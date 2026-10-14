using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.RLBridge
{
    internal static class AutonomousHistoryBridge
    {
        private static AutonomousElderHistoryEnvironment environment;
        internal static object Handle(string op, JsonElement root, JsonSerializerOptions json)
        {
            if (op == "autonomous5_schema") return new { ok = true, actionVersion = 3,
                observationVersion = 5, environmentVersion = 5, replayVersion = 5,
                roles = WorkforceRoles.All, activities = AutonomousElderContract.ActivityNames,
                goods = Enum.GetNames(typeof(TradeGood)), contextWidth = 512, contacts = 8,
                historyWidth = 300, featureWidth = 1061, historyFeatures = AutonomousElderHistoryContract.FeatureNames,
                saveEnvelope = "ANTHROPOCENE-12" };
            if (op == "autonomous5_reset")
            {
                var snapshot = Convert.FromBase64String(root.GetProperty("snapshot").GetString());
                var ids = root.GetProperty("controlled").Deserialize<int[]>(json);
                var candidate = new AutonomousElderHistoryEnvironment();
                var observations = candidate.Reset(snapshot, ids, root.GetProperty("decisionLimit").GetInt32());
                environment = candidate;
                return new { ok = true, observations, fingerprint = environment.Replay.InitialFingerprint,
                    buildHash = environment.Replay.BuildHash, configurationHash = environment.Replay.ConfigurationHash };
            }
            if (environment == null) throw new InvalidOperationException("autonomous5_reset required.");
            if (op == "autonomous5_step")
            {
                var actions = root.GetProperty("actions").Deserialize<Dictionary<int, AutonomousElderAction>>(json);
                var timer = Stopwatch.StartNew(); var result = environment.Step(actions); timer.Stop();
                return new { ok = true, result, nativeStepSeconds = timer.Elapsed.TotalSeconds };
            }
            if (op == "autonomous5_verify_replay")
            { AutonomousElderHistoryEnvironment.VerifyReplay(environment.Replay); return new { ok = true, steps = environment.Replay.Steps.Count }; }
            if (op == "autonomous5_capture") return new { ok = true, snapshot = environment.CaptureSnapshot(), replay = environment.Replay };
            throw new ArgumentException("Unknown autonomous5 operation.");
        }
    }
}
