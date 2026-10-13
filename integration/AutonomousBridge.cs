using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.RLBridge
{
    internal static class AutonomousBridge
    {
        private static AutonomousElderEnvironment environment;
        internal static object Handle(string op, JsonElement root, JsonSerializerOptions json)
        {
            if (op == "autonomous_schema") return new { ok = true, actionVersion = AutonomousElderContract.ActionVersion,
                observationVersion = AutonomousElderContract.ObservationVersion, environmentVersion = AutonomousElderContract.EnvironmentVersion,
                replayVersion = AutonomousElderContract.ReplayVersion, roles = WorkforceRoles.All, activities = AutonomousElderContract.ActivityNames,
                goods = Enum.GetNames(typeof(TradeGood)), contextWidth = 512, contacts = AutonomousElderContract.Contacts };
            if (op == "autonomous_reset")
            {
                var snapshot = Convert.FromBase64String(root.GetProperty("snapshot").GetString());
                var ids = root.GetProperty("controlled").Deserialize<int[]>(json);
                int limit = root.GetProperty("decisionLimit").GetInt32();
                var candidate = new AutonomousElderEnvironment();
                var observations = candidate.Reset(snapshot, ids, limit); environment = candidate;
                return new { ok = true, observations, fingerprint = environment.Replay.InitialFingerprint, buildHash = environment.Replay.BuildHash, configurationHash = environment.Replay.ConfigurationHash };
            }
            if (environment == null) throw new InvalidOperationException("autonomous_reset required.");
            if (op == "autonomous_step")
            {
                var actions = root.GetProperty("actions").Deserialize<Dictionary<int, AutonomousElderAction>>(json);
                var timer = Stopwatch.StartNew(); var result = environment.Step(actions); timer.Stop();
                return new { ok = true, result, nativeStepSeconds = timer.Elapsed.TotalSeconds };
            }
            if (op == "autonomous_verify_replay")
            {
                AutonomousElderEnvironment.VerifyReplay(environment.Replay);
                return new { ok = true, steps = environment.Replay.Steps.Count };
            }
            if (op == "autonomous_capture") return new { ok = true, snapshot = environment.CaptureSnapshot(), replay = environment.Replay };
            throw new ArgumentException("Unknown autonomous op.");
        }
    }
}
