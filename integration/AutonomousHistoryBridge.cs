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
        private static AutonomousElderHistoryEnvironment growthEnvironment;
        internal static object Handle(string op, JsonElement root, JsonSerializerOptions json)
        {
            bool growth = op.StartsWith("autonomous6_", StringComparison.Ordinal);
            var active = growth ? growthEnvironment : environment;
            if (op == "autonomous6_schema") return new { ok = true, actionVersion = 4,
                observationVersion = 6, environmentVersion = 6, replayVersion = 6,
                roles = WorkforceRoles.All, activities = AutonomousElderContract.ActivityNames,
                goods = Enum.GetNames(typeof(TradeGood)), contextWidth = 512, contacts = 8,
                historyWidth = 301, featureWidth = 1062,
                historyFeatures = System.Linq.Enumerable.Concat(AutonomousElderHistoryContract.FeatureNames,
                    new[] { "previous_action_version" }), saveEnvelope = "ANTHROPOCENE-13",
                additionalSaveEnvelopes = new[] { "ANTHROPOCENE-14" },
                needsProfiles = new[] { "legacy", "subsistence-needs-v13" },
                controlTransform = "clip((latent-.05)/.90,0,1)", herdRetention = "post-growth" };
            if (growth) op = op.Replace("autonomous6_", "autonomous5_");
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
                var candidate = new AutonomousElderHistoryEnvironment(growthAuthority: growth);
                var observations = candidate.Reset(snapshot, ids, root.GetProperty("decisionLimit").GetInt32());
                if (growth) growthEnvironment = candidate; else environment = candidate;
                return new { ok = true, observations, fingerprint = candidate.Replay.InitialFingerprint,
                    buildHash = candidate.Replay.BuildHash, configurationHash = candidate.Replay.ConfigurationHash };
            }
            if (op == "autonomous5_prepare_probe")
            {
                // Pure derivation does not require or mutate an active episode.
                var snapshot = Convert.FromBase64String(root.GetProperty("snapshot").GetString());
                int camp = root.GetProperty("focal").GetInt32();
                string transform = root.GetProperty("transform").GetString();
                if (transform == "subsistence_needs_v13" && !growth)
                    throw new ArgumentException("Native6 is required for the subsistence needs profile.");
                return new { ok = true, snapshot = AutonomousOwnerDiagnostics.Derive(snapshot, camp, transform) };
            }
            if (active == null) throw new InvalidOperationException("Versioned autonomous reset required.");
            if (op == "autonomous5_quote_address")
                return new { ok = true, metadataVersion = "own-known-address-v1",
                    ordinal = active.KnownCounterpartyOrdinal(root.GetProperty("observer").GetInt32(),
                        root.GetProperty("target").GetInt32()) };
            if (op == "autonomous5_step")
            {
                var actions = root.GetProperty("actions").Deserialize<Dictionary<int, AutonomousElderAction>>(json);
                var timer = Stopwatch.StartNew(); var result = active.Step(actions); timer.Stop();
                if (growth && root.TryGetProperty("outcomeDiagnostics", out var flag) && flag.GetBoolean())
                    return new { ok = true, result, nativeStepSeconds = timer.Elapsed.TotalSeconds,
                        outcomeDiagnostics = System.Linq.Enumerable.ToDictionary(actions.Keys, id => id,
                            id => active.ClosedOutcomeDiagnostics(id)) };
                return new { ok = true, result, nativeStepSeconds = timer.Elapsed.TotalSeconds };
            }
            if (op == "autonomous5_verify_replay")
            { AutonomousElderHistoryEnvironment.VerifyReplay(active.Replay); return new { ok = true, steps = active.Replay.Steps.Count }; }
            if (op == "autonomous5_capture")
            {
                var snapshot = active.CaptureSnapshot();
                if (root.TryGetProperty("ownerDiagnostics", out var diagnostics) && diagnostics.GetBoolean())
                    return new { ok = true, snapshot, replay = active.Replay,
                        ownerDiagnostics = AutonomousOwnerDiagnostics.Capture(snapshot, active.Replay.Controlled) };
                return new { ok = true, snapshot, replay = active.Replay };
            }
            throw new ArgumentException("Unknown autonomous5 operation.");
        }
    }
}
