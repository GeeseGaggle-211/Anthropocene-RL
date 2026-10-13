using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.HeuristicAudit
{
    // Diagnostic executable only. Native planning, economics and reward code are reused.
    internal sealed class Session
    {
        private WorldState world;
        private int focal, peer, limit, steps;
        private byte[] initial;
        private bool ended;
        private readonly PhaseExecution execution = new PhaseExecution();
        private readonly List<Dictionary<int, AutonomousElderAction>> actions = new List<Dictionary<int, AutonomousElderAction>>();
        private readonly List<string> fingerprints = new List<string>();
        internal object Reset(byte[] snapshot, int focalId, int peerId, int decisionLimit)
        {
            if (decisionLimit < 1 || decisionLimit > 25 || focalId == peerId) throw new ArgumentException("Invalid audit boundary.");
            world = SaveCodec.Load((byte[])snapshot.Clone());
            if (!world.Camps.Any(c => c.Id == focalId && c.Population > 0)) throw new ArgumentException("Missing focal village.");
            focal = focalId; peer = peerId; limit = decisionLimit; steps = 0; ended = false;
            initial = SaveCodec.Save(world); actions.Clear(); fingerprints.Clear();
            return new { ok = true, observations = Observe(), fingerprint = SaveCodec.Fingerprint(world),
                buildHash = NativeElderEnvironment.BuildHash, configurationHash = SaveCodec.ConfigurationHash(world.Config),
                policy = "AiPlanning.Decide + TurnPlanner.Prepare native economic planner", economicAllocation = world.Config.EconomicAllocation.ToString(),
                peerMode = peer < 0 ? "native" : "public-market" };
        }
        private Dictionary<int, AutonomousElderObservation> Observe()
            => world.Camps.Where(c => c.Population > 0 && (c.Id == focal || c.Id == peer)).OrderBy(c => c.Id)
                .ToDictionary(c => c.Id, c => AutonomousElderCompiler.Observe(world, c));
        internal object Step(Dictionary<int, AutonomousElderAction> submitted)
        {
            if (world == null || ended) throw new InvalidOperationException("Reset required.");
            var timer = Stopwatch.StartNew();
            var obs = Observe();
            var expected = peer >= 0 && obs.ContainsKey(peer) ? new[] { peer } : Array.Empty<int>();
            if (!submitted.Keys.OrderBy(x => x).SequenceEqual(expected)) throw new ArgumentException("Only the configured living market peer may receive an action.");
            foreach (var pair in submitted) AutonomousElderCompiler.Validate(pair.Value, obs[pair.Key]);
            var requested = submitted.ToDictionary(x => x.Key, x => x.Value.Copy());
            var previous = world.Camps.FirstOrDefault(c => c.Id == focal);
            int population = previous?.Population ?? 0, tile = previous?.Tile ?? -1;
            var orders = AiPlanning.Decide(world, execution).Orders.ToDictionary(x => x.Key,
                x => x.Value.Where(o => !requested.ContainsKey(o.Entity)).Select(o => o.Copy()).ToList());
            foreach (var camp in world.Camps.Where(c => requested.ContainsKey(c.Id)))
            {
                if (!orders.ContainsKey(camp.Player)) orders[camp.Player] = new List<Order>();
                int move = requested[camp.Id].Movement;
                orders[camp.Player].Add(move == 0 ? new Order(OrderKind.NoOp, camp.Id) :
                    new Order(OrderKind.Move, camp.Id, destination: obs[camp.Id].SiteTiles[move - 1]));
            }
            var prepared = requested.Count == 0 ? TurnPlanner.Prepare(world, orders, execution: execution) :
                TurnPlanner.PrepareAutonomous(world, orders, requested, obs, execution);
            var record = new TurnResolver(execution).Resolve(world, prepared);
            steps++;
            var current = world.Camps.FirstOrDefault(c => c.Id == focal);
            double years = record.Duration / world.Config.DaysPerYear;
            var observations = Observe();
            bool terminated = current == null || current.Population <= 0;
            bool truncated = !terminated && steps >= limit;
            ended = terminated || truncated;
            string fingerprint = SaveCodec.Fingerprint(world);
            actions.Add(requested); fingerprints.Add(fingerprint);
            double quality = current == null ? 0 : ElderRLContract.Unit(current.QoL);
            var telemetry = new AutonomousElderTelemetry {
                Population = new Dictionary<int, int> { [focal] = current?.Population ?? 0 },
                Quality = new Dictionary<int, double> { [focal] = quality },
                FoodMet = new Dictionary<int, double> { [focal] = current?.FoodNeedsMetRatio ?? 0 },
                Moved = new Dictionary<int, int> { [focal] = current != null && current.Tile != tile ? 1 : 0 },
                SettledTrades = new Dictionary<int, int> { [focal] = record.Trades.Count(t => t.Requester == focal || t.Responder == focal) },
                DaughterSettlements = new Dictionary<int, int> { [focal] = record.Fissions.Count(f => f.ParentCamp == focal) },
                Immigrated = new Dictionary<int, int> { [focal] = record.Migrations.Where(m => m.DestinationCamp == focal).Sum(m => m.Migrants) },
                Emigrated = new Dictionary<int, int> { [focal] = record.Migrations.Where(m => m.SourceCamp == focal).Sum(m => m.Migrants) },
                Trades = record.Trades.Count, Fissions = record.Fissions.Count, Migrants = record.Migrations.Sum(m => m.Migrants)
            };
            var result = new AutonomousElderStep { Observations = observations,
                Rewards = new Dictionary<int, double> { [focal] = ElderRewardContract.Physical(population, quality, years).UndiscountedIntervalWelfare },
                Staffing = new Dictionary<int, int[]>(), Telemetry = telemetry, ElapsedYears = years,
                Terminated = terminated, Truncated = truncated, Fingerprint = fingerprint };
            timer.Stop();
            return new { ok = true, result, nativeStepSeconds = timer.Elapsed.TotalSeconds };
        }
        internal object Verify()
        {
            var replay = new Session(); replay.Reset(initial, focal, peer, limit);
            for (int i = 0; i < actions.Count; i++)
            {
                replay.Step(actions[i]);
                if (replay.fingerprints[i] != fingerprints[i]) throw new InvalidOperationException("Heuristic replay divergence.");
            }
            if (SaveCodec.Fingerprint(SaveCodec.Load(SaveCodec.Save(world))) != SaveCodec.Fingerprint(world))
                throw new InvalidOperationException("Heuristic save/load divergence.");
            return new { ok = true, steps = steps, fingerprint = SaveCodec.Fingerprint(world) };
        }
    }

    internal static class Program
    {
        private static readonly JsonSerializerOptions Json = new JsonSerializerOptions { IncludeFields = true, PropertyNameCaseInsensitive = true };
        private static int Main()
        {
            var session = new Session(); string line;
            while ((line = Console.ReadLine()) != null)
            {
                try
                {
                    using var document = JsonDocument.Parse(line); var root = document.RootElement;
                    string op = root.GetProperty("op").GetString(); object result;
                    if (op == "close") { Console.WriteLine("{\"ok\":true}"); return 0; }
                    if (op == "heuristic_reset") result = session.Reset(Convert.FromBase64String(root.GetProperty("snapshot").GetString()),
                        root.GetProperty("focal").GetInt32(), root.GetProperty("peer").GetInt32(), root.GetProperty("decisionLimit").GetInt32());
                    else if (op == "heuristic_step") result = session.Step(root.GetProperty("actions").Deserialize<Dictionary<int, AutonomousElderAction>>(Json));
                    else if (op == "heuristic_verify") result = session.Verify();
                    else throw new ArgumentException("Unknown heuristic audit operation.");
                    Console.WriteLine(JsonSerializer.Serialize(result, Json));
                }
                catch (Exception error) { Console.WriteLine(JsonSerializer.Serialize(new { ok = false, error = error.Message }, Json)); }
            }
            return 0;
        }
    }
}
