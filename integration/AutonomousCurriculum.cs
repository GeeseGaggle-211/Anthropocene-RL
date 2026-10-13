using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text.Json;
using System.Security.Cryptography;
using Anthropocene.Simulation;
namespace Anthropocene.SnapshotBank
{
    internal static class AutonomousCurriculum
    {
        static void Main(string[] args)
        {
            var json = new JsonSerializerOptions { IncludeFields = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
            var plan = JsonSerializer.Deserialize<GenerationPlan>(File.ReadAllText(args[0]), json);
            if (Directory.Exists(args[1])) throw new ArgumentException("Output must be new; preserve previous banks.");
            Directory.CreateDirectory(args[1]);
            var factory = new SnapshotScenarioFactory(plan); var entries = new List<object>();
            foreach (var t in plan.Trajectories)
            {
                var result = factory.Build(t); var w = result.World;
                w.Config.TradeFoodEnabled = true;
                if (!w.Config.FissionEnabled || !w.Config.ImmigrationEnabled || !w.Config.TradeEnabled) throw new Exception("All dynamics must remain enabled.");
                var c = w.Camps.Single(x => x.Id == result.ControlledCampId);
                // Only the known reciprocal F15 market partner is scripted. All other
                // settlements (including daughters) retain native background decisions.
                var peer = t.ScenarioFamily == "F15" ? w.Camps.Where(x => x.Player != 0 && x.Player != c.Player).OrderBy(x => x.Id).First() : null;
                byte[] snapshot = SaveCodec.Save(w);
                string file = t.TrajectoryId + ".anthro"; File.WriteAllBytes(Path.Combine(args[1], file), snapshot);
                entries.Add(new { id = t.TrajectoryId, family = t.TrajectoryFamily, partition = t.Partition, worldSeed = t.WorldSeed,
                    snapshot = file, sha256 = Convert.ToHexString(SHA256.HashData(snapshot)).ToLowerInvariant(), fingerprint = SaveCodec.Fingerprint(w),
                    focal = c.Id, peer = peer?.Id, population = c.Population, configurationHash = SaveCodec.ConfigurationHash(w.Config),
                    dynamics = new { movement = true, trade = w.Config.TradeEnabled, migration = w.Config.ImmigrationEnabled, fission = w.Config.FissionEnabled } });
                Console.WriteLine(t.TrajectoryId);
            }
            File.WriteAllText(Path.Combine(args[1], "manifest.json"), JsonSerializer.Serialize(new { version = "autonomous-curriculum-v1", entries }, json));
        }
    }
}
