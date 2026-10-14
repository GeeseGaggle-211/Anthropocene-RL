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
    internal static class AutonomousOpportunityCurriculum
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
            int step, string family, bool peer, int targetOrdinal, int focalGive)
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
                int give = peer ? 9 - focalGive : focalGive, want = 9 - give;
                int partner = targetOrdinal;
                if (partner >= 0 && o.CounterpartyMask[partner] && o.GoodMask[give] && o.GoodMask[want] && o.Stocks[give] >= 1e-6)
                { double q = Math.Max(1e-6, Math.Min(1e6, o.Stocks[give] * .05));
                    a.Quote = new ElderStockQuote { Counterparty = partner, Give = (TradeGood)give, Want = (TradeGood)want, GiveQuantity = q, WantQuantity = q }; }
            }
            if (!peer && family == "migration" && step == 0)
                a.Movement = Array.FindIndex(o.MovementMask, 1, x => x) is int m && m >= 0 ? m : 0;
            AutonomousElderCompiler.Validate(a, o);
            return a;
        }

        private static void Main(string[] args)
        {
            if (args.Length != 2 || Directory.Exists(args[1]))
                throw new ArgumentException("Frozen plan and fresh output directory required.");
            var raw = File.ReadAllBytes(args[0]); var plan = JsonSerializer.Deserialize<GenerationPlan>(raw, Json);
            if (plan.Trajectories.Length != 48 || plan.Trajectories.Any(t => !t.SnapshotSteps.SequenceEqual(new[] { 0, 2, 8, 25 })))
                throw new ArgumentException("Frozen48-trajectory,25-decision generation envelope required.");
            if (plan.BankKind != "autonomous-opportunity-v9" || plan.World.Players != 2 ||
                plan.Trajectories.GroupBy(t => t.WorldSeed).Count() != 8 ||
                plan.Trajectories.GroupBy(t => t.WorldSeed).Any(g => g.Count() != 6 ||
                    g.Select(t => t.Partition).Distinct().Count() != 1 ||
                    !g.Select(t => t.TrajectoryFamily).OrderBy(x => x).SequenceEqual(
                        new[] { "fishing", "fission", "husbandry", "migration", "survival", "trade" })))
                throw new ArgumentException("Eight world-disjoint six-family cohorts required.");
            Directory.CreateDirectory(args[1]); var clock = Stopwatch.StartNew();
            var factory = new SnapshotScenarioFactory(plan); var entries = new List<object>(); var trajectories = new List<object>();
            var unavailable = new List<object>(); var failures = new List<object>();
            var completed = new HashSet<string>();
            string active = null; int activeStep = 0;
            int generation = 0, replay = 0; double priorWall = 0;
            double Wall() => priorWall + clock.Elapsed.TotalSeconds;
            if (priorWall >= 600 || generation >= 1200 || replay >= 1200) throw new TimeoutException("Cumulative generation envelope exhausted.");
            using var wallTimer = new System.Threading.Timer(_ => { Console.Error.WriteLine("Generation wall cap reached; retain manifest and progress ledger.");
                System.Environment.Exit(124); }, null, TimeSpan.FromSeconds(600 - priorWall), System.Threading.Timeout.InfiniteTimeSpan);
            void Save(string status) => File.WriteAllText(Path.Combine(args[1], "manifest.json"), JsonSerializer.Serialize(new {
                version = "autonomous-opportunity-v9", status, action_version = 4, observation_version = 6,
                environment_version = 6, replay_version = 6, save_envelope = "ANTHROPOCENE-13",
                source_plan_sha256 = Sha(raw), source_controller = "scripted-native-demand-control-action4-known-customer",
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
                    if (w.Camps.Count != 2) throw new InvalidOperationException("Exactly two initial settlements required.");
                    if (!w.Config.TradeEnabled || !w.Config.ImmigrationEnabled || !w.Config.FissionEnabled)
                        throw new InvalidOperationException("Movement/trade/migration/fission remain intrinsic.");
                    int focal = built.ControlledCampId;
                    int? peer = t.ScenarioFamily == "F15" ? w.Camps.Where(c => c.Id != focal)
                        .OrderBy(c => c.Id).Select(c => (int?)c.Id).FirstOrDefault() : null;
                    var ids = peer.HasValue ? new[] { focal, peer.Value } : new[] { focal };
                    var env = new AutonomousElderHistoryEnvironment(growthAuthority: true); var obs = env.Reset(SaveCodec.Save(w), ids, 25);
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
                            entries.Add(new { id, snapshot = file, sha256 = Sha(bytes), fingerprint = SaveCodec.Fingerprint(state),
                                world_seed = t.WorldSeed, partition = t.Partition, lineage_id = t.LineageId,
                                counterfactual_group_id = t.CounterfactualGroupId, family = t.TrajectoryFamily,
                                focal, peer, population = c.Population, decision_index = steps,
                                owned_animals = state.Herds.Where(h => h.DomesticCamp == focal).Sum(h => h.Animals),
                                market_focal_give = t.CausalVariant == "surplus-export" ? 3 : 6,
                                institution_age_years = obs[focal].History[299], prior_action_present = obs[focal].History[0] == 1,
                                history_maturity = obs[focal].Context.Skip(117).Take(3).ToArray(),
                                configuration_hash = SaveCodec.ConfigurationHash(state.Config), native_build_hash = NativeElderEnvironment.BuildHash,
                                neural_training_eligible = true,
                                eligibility_note = "action4 mechanics proved; v9 opportunity/mask admission audit required",
                                workforce_mask = obs[focal].WorkforceMask, control_mask = obs[focal].ControlMask,
                                dynamics = new { movement = true, trade = true, migration = true, fission = true } });
                        }
                        if (steps >= 25 || !obs.ContainsKey(focal)) break;
                        if (Wall() > 600 || generation >= 1200) throw new TimeoutException("Generation envelope exhausted.");
                        var state0 = SaveCodec.Load(env.CaptureSnapshot());
                        var actions = obs.ToDictionary(p => p.Key, p => Policy(state0, state0.Camps.Single(c => c.Id == p.Key), p.Value,
                            steps, t.TrajectoryFamily, p.Key != focal, ids.Length == 2 ? env.KnownCounterpartyOrdinal(p.Key, p.Key == focal ? ids[1] : focal) : -1, t.CausalVariant == "surplus-export" ? 3 : 6));
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
                    if (replay > 1200) throw new InvalidOperationException("Replay envelope exhausted.");
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
