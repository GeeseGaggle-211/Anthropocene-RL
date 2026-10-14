using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.SnapshotBank
{
    // Source-controlled native scenario generation, with explicitly scripted
    // current-interface actions. No learner/model/optimizer is involved.
    internal static class AutonomousHistoryCurriculum
    {
        private static readonly JsonSerializerOptions Json = new JsonSerializerOptions {
            IncludeFields = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
        private static string Sha(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        private static bool UnavailableSite(string message) =>
            message == "No physically compatible site for requested fishing/field strata." ||
            message == "Requested domestic herd support ratio exceeds the native maximum herd size." ||
            message == "Requested domestic herd has zero local sustainable support." ||
            message == "Requested domestic herd support ratio is below the minimum whole-animal fixture size." ||
            message == "Native Fishing access did not reach the requested number of grounds." ||
            message == "No physically compatible open-economy partner site.";
        private static AutonomousElderAction Policy(WorldState w, Camp c, AutonomousElderHistoryObservation o,
            int step, string family, bool peer)
        {
            var a = AutonomousElderCompiler.Uniform(o);
            var plan = EconomicAllocationStrategy.Plan(w, c,
                EconomicIntent.From(c), out _);
            var counts = new int[13];
            foreach (var x in plan.Assignments)
            { int role = Array.IndexOf(WorkforceRoles.All, x.Role); if (role >= 0 && o.WorkforceMask[role]) counts[role] += x.Population; }
            int total = counts.Sum();
            if (total < c.Population) counts[0] += c.Population - total;
            double denominator = counts.Sum();
            a.Workforce = counts.Select(n => n / denominator).ToArray();
            if (o.ActivityMask[0][0])
            {
                double materialShare = counts[0] == 0 ? 0 : Math.Min(1, Math.Max(0, plan.Labor?.MaterialGatherers ?? 0) / counts[0]);
                Array.Clear(a.Activities[0], 0, 15); a.Activities[0][0] = 1 - materialShare;
                for (int i = 1; i <= 3; i++) a.Activities[0][i] = materialShare / 3;
            }
            a.FieldExpansionFraction = o.ControlMask[0] && counts[1] > 0 ?
                Math.Min(1, Math.Max(0, plan.Labor?.ExpansionFarmers ?? 0) / counts[1]) : 0;
            a.Breeding = o.ControlMask[2] ? plan.ManagedBreedingIntensity : 0;
            a.RetainedHerdFraction = o.ControlMask[1] ? 1 : 0;
            // Fixed public quote: own stock and known contact only. Both parties
            // independently quote the same1:1 terms; escrow may partially fill it.
            if (family == "trade" || family == "migration")
            {
                int give = peer ? 3 : 6, want = peer ? 6 : 3;
                int partner = Array.FindIndex(o.CounterpartyMask, x => x);
                if (partner >= 0 && o.GoodMask[give] && o.GoodMask[want] && o.Stocks[give] >= 1e-6)
                { double q = Math.Max(1e-6, Math.Min(10, o.Stocks[give]));
                    a.Quote = new ElderStockQuote { Counterparty = partner, Give = (TradeGood)give, Want = (TradeGood)want, GiveQuantity = q, WantQuantity = q }; }
            }
            if (!peer && family == "migration" && step == 0)
                a.Movement = Array.FindIndex(o.MovementMask, 1, x => x) is int m && m >= 0 ? m : 0;
            AutonomousElderCompiler.Validate(a, o);
            return a;
        }

        private static void Main(string[] args)
        {
            bool resume = args.Length == 3 && args[2] == "--resume";
            if ((!resume && args.Length != 2) || (resume != Directory.Exists(args[1])))
                throw new ArgumentException("Plan and new output directory required, or an existing failed bank with --resume.");
            var raw = File.ReadAllBytes(args[0]); var plan = JsonSerializer.Deserialize<GenerationPlan>(raw, Json);
            if (plan.Trajectories.Length != 144 || plan.Trajectories.Any(t => !t.SnapshotSteps.SequenceEqual(new[] { 0, 2, 8, 25 })))
                throw new ArgumentException("Frozen144-trajectory,25-decision generation envelope required.");
            Directory.CreateDirectory(args[1]); var clock = Stopwatch.StartNew();
            var factory = new SnapshotScenarioFactory(plan); var entries = new List<object>(); var trajectories = new List<object>();
            var unavailable = new List<object>(); var failures = new List<object>();
            var completed = new HashSet<string>();
            string active = null; int activeStep = 0;
            int generation = 0, replay = 0; double priorWall = 0;
            if (resume)
            {
                string manifestPath = Path.Combine(args[1], "manifest.json");
                var prior = JsonDocument.Parse(File.ReadAllBytes(manifestPath)).RootElement;
                if (prior.GetProperty("status").GetString() != "FAILED" || prior.GetProperty("source_plan_sha256").GetString() != Sha(raw))
                    throw new ArgumentException("Resume requires the same frozen plan and a failed manifest.");
                foreach (var x in prior.GetProperty("failures").EnumerateArray())
                    if (!UnavailableSite(x.GetProperty("error").GetString().Split('\n')[0].Replace("System.InvalidOperationException: ", "")))
                        throw new ArgumentException("Only the known physical-site gap can be resumed without a new gate.");
                int attempt = 1;
                while (File.Exists(Path.Combine(args[1], "manifest.attempt" + attempt + ".json"))) attempt++;
                string preserved = Path.Combine(args[1], "manifest.attempt" + attempt + ".json");
                File.Copy(manifestPath, preserved);
                generation = prior.GetProperty("generation_resolves").GetInt32(); replay = prior.GetProperty("replay_resolves").GetInt32();
                priorWall = prior.GetProperty("wall_seconds").GetDouble();
                foreach (var x in prior.GetProperty("entries").EnumerateArray())
                {
                    if (Sha(File.ReadAllBytes(Path.Combine(args[1], x.GetProperty("snapshot").GetString()))) != x.GetProperty("sha256").GetString())
                        throw new InvalidOperationException("Preserved snapshot hash mismatch.");
                    entries.Add(x.Clone());
                }
                foreach (var x in prior.GetProperty("trajectories").EnumerateArray())
                { trajectories.Add(x.Clone()); completed.Add(x.GetProperty("id").GetString()); }
                foreach (var x in prior.GetProperty("unavailable").EnumerateArray())
                { unavailable.Add(x.Clone()); completed.Add(x.GetProperty("id").GetString()); }
                foreach (var x in prior.GetProperty("failures").EnumerateArray())
                {
                    failures.Add(x.Clone()); string id = x.GetProperty("trajectory").GetString();
                    completed.Add(id);
                    foreach (int endpoint in plan.Trajectories.Single(t => t.TrajectoryId == id).SnapshotSteps)
                        if (!prior.GetProperty("unavailable").EnumerateArray().Any(u =>
                            u.GetProperty("id").GetString() == id && u.GetProperty("decision_index").GetInt32() == endpoint))
                            unavailable.Add(new { id, decision_index = endpoint, ended_at_decision = 0,
                                reason = x.GetProperty("error").GetString().Split('\n')[0] });
                }
            }
            double Wall() => priorWall + clock.Elapsed.TotalSeconds;
            if (priorWall >= 600 || generation >= 3600 || replay >= 3600) throw new TimeoutException("Cumulative generation envelope exhausted.");
            using var wallTimer = new System.Threading.Timer(_ => { Console.Error.WriteLine("Generation wall cap reached; retain manifest and progress ledger.");
                System.Environment.Exit(124); }, null, TimeSpan.FromSeconds(600 - priorWall), System.Threading.Timeout.InfiniteTimeSpan);
            void Save(string status) => File.WriteAllText(Path.Combine(args[1], "manifest.json"), JsonSerializer.Serialize(new {
                version = "autonomous-history-fresh-v5", status, action_version = 3, observation_version = 5,
                environment_version = 5, replay_version = 5, save_envelope = "ANTHROPOCENE-12",
                source_plan_sha256 = Sha(raw), source_controller = "scripted-native-demand-control-projected",
                retained_native_training = 0, generation_resolves = generation, replay_resolves = replay,
                wall_seconds = Wall(), entries, trajectories, unavailable, failures }, Json));
            void Progress() => File.WriteAllText(Path.Combine(args[1], "progress.json"), JsonSerializer.Serialize(new {
                trajectory = active, step = activeStep, generation_resolves = generation,
                replay_resolves = replay, wall_seconds = Wall() }, Json));
            Save("RUNNING");
            try
            {
                foreach (var t in plan.Trajectories)
                {
                    if (completed.Contains(t.TrajectoryId)) continue;
                    active = t.TrajectoryId; activeStep = 0; var captured = new HashSet<int>();
                    if (Wall() > 600) throw new TimeoutException("Native generation600-second cap.");
                    ScenarioBuildResult built;
                    try { built = factory.Build(t); }
                    catch (InvalidOperationException error) when (UnavailableSite(error.Message))
                    {
                        foreach (int endpoint in t.SnapshotSteps) unavailable.Add(new { id = t.TrajectoryId,
                            decision_index = endpoint, ended_at_decision = 0, reason = error.Message });
                        Save("RUNNING"); Console.WriteLine(t.TrajectoryId + " UNAVAILABLE: " + error.Message); continue;
                    }
                    var w = built.World; w.Config.TradeFoodEnabled = true;
                    if (!w.Config.TradeEnabled || !w.Config.ImmigrationEnabled || !w.Config.FissionEnabled)
                        throw new InvalidOperationException("Movement/trade/migration/fission remain intrinsic.");
                    int focal = built.ControlledCampId;
                    int? peer = t.ScenarioFamily == "F15" ? w.Camps.Where(c => c.Player != 0 && c.Id != focal)
                        .OrderBy(c => c.Id).Select(c => (int?)c.Id).FirstOrDefault() : null;
                    var ids = peer.HasValue ? new[] { focal, peer.Value } : new[] { focal };
                    var env = new AutonomousElderHistoryEnvironment(); var obs = env.Reset(SaveCodec.Save(w), ids, 25);
                    int steps = 0, trades = 0, moves = 0, migrants = 0, daughters = 0;
                    while (true)
                    {
                        if (t.SnapshotSteps.Contains(steps) && obs.ContainsKey(focal))
                        {
                            string committed = steps == 0 ? env.Replay.InitialFingerprint : env.Replay.Steps.Last().Fingerprint;
                            var bytes = env.CaptureSnapshot(); var state = SaveCodec.Load(bytes); var c = state.Camps.Single(c => c.Id == focal);
                            string id = t.TrajectoryId + "-d" + steps, file = id + ".anthro";
                            string target = Path.Combine(args[1], file);
                            if (File.Exists(target)) throw new InvalidOperationException("Snapshot already exists; never overwrite source data.");
                            File.WriteAllBytes(target, bytes);
                            if (SaveCodec.Fingerprint(state) != committed)
                                throw new InvalidOperationException("Native snapshot roundtrip mismatch.");
                            captured.Add(steps);
                            bool coreFace = t.Economy == "cultivation" && !obs[focal].WorkforceMask[2] &&
                                !obs[focal].WorkforceMask[3] && !obs[focal].ControlMask[1] && !obs[focal].ControlMask[2];
                            entries.Add(new { id, snapshot = file, sha256 = Sha(bytes), fingerprint = SaveCodec.Fingerprint(state),
                                world_seed = t.WorldSeed, partition = t.Partition, lineage_id = t.LineageId,
                                counterfactual_group_id = t.CounterfactualGroupId, family = t.TrajectoryFamily,
                                focal, peer, population = c.Population, decision_index = steps,
                                institution_age_years = obs[focal].History[299], prior_action_present = obs[focal].History[0] == 1,
                                history_maturity = obs[focal].Context.Skip(117).Take(3).ToArray(),
                                configuration_hash = SaveCodec.ConfigurationHash(state.Config), native_build_hash = NativeElderEnvironment.BuildHash,
                                neural_training_eligible = coreFace,
                                eligibility_note = coreFace ? "prospective core cohort, pending independent audit" : "actual current action-face evidence quarantine",
                                workforce_mask = obs[focal].WorkforceMask, control_mask = obs[focal].ControlMask,
                                dynamics = new { movement = true, trade = true, migration = true, fission = true } });
                        }
                        if (steps >= 25 || !obs.ContainsKey(focal)) break;
                        if (Wall() > 600 || generation >= 3600) throw new TimeoutException("Generation envelope exhausted.");
                        var state0 = SaveCodec.Load(env.CaptureSnapshot());
                        var actions = obs.ToDictionary(p => p.Key, p => Policy(state0, state0.Camps.Single(c => c.Id == p.Key), p.Value,
                            steps, t.TrajectoryFamily, p.Key != focal));
                        generation++; activeStep = steps; Progress();
                        var r = env.Step(actions); steps++; obs = r.Observations;
                        trades += r.Telemetry.SettledTrades[focal]; moves += r.Telemetry.Moved[focal];
                        migrants += r.Telemetry.Immigrated[focal] + r.Telemetry.Emigrated[focal]; daughters += r.Telemetry.DaughterSettlements[focal];
                        if (r.Terminated) break;
                    }
                    foreach (int endpoint in t.SnapshotSteps.Where(s => !captured.Contains(s)))
                        unavailable.Add(new { id = t.TrajectoryId, decision_index = endpoint,
                            ended_at_decision = steps, reason = "focal village no longer living/observable" });
                    replay += steps; Progress();
                    if (replay > 3600) throw new InvalidOperationException("Replay envelope exhausted.");
                    AutonomousElderHistoryEnvironment.VerifyReplay(env.Replay, new PhaseExecution(4, 1, true, 1));
                    trajectories.Add(new { id = t.TrajectoryId, partition = t.Partition, family = t.TrajectoryFamily,
                        world_seed = t.WorldSeed, steps, trades, moves, migrants, daughters, replay = "PASS" });
                    Save("RUNNING"); Console.WriteLine(t.TrajectoryId);
                }
                Save(unavailable.Count == 0 ? "PASS" : "PASS_WITH_GAPS");
            }
            catch (Exception error) { failures.Add(new { trajectory = active, step = activeStep, error = error.ToString() }); Save("FAILED"); throw; }
        }
    }
}
