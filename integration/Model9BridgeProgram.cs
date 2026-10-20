using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Anthropocene.Simulation;

// JSON-lines bridge for native Elder contract 8 (model9). One environment per process; the learner never
// reimplements physics. Ops: schema, reset {seed, players, movement} | {snapshot}, step {actions}, verify_replay,
// capture, close. Missing actions run the native heuristic for that village.
namespace Anthropocene.RLBridge
{
    internal static class Model9BridgeProgram
    {
        private static readonly JsonSerializerOptions Json = new JsonSerializerOptions
        {
            IncludeFields = true, PropertyNameCaseInsensitive = true, WriteIndented = false
        };
        private static ElderModel9Environment environment;

        private static int Main()
        {
            string line;
            while ((line = Console.ReadLine()) != null)
            {
                try
                {
                    using var document = JsonDocument.Parse(line);
                    var root = document.RootElement;
                    string op = root.GetProperty("op").GetString() ?? "";
                    if (op == "close") { Write(new { ok = true, op }); return 0; }
                    Write(Handle(op, root));
                }
                catch (Exception error)
                {
                    Write(new { ok = false, error = error.GetType().Name + ": " + error.Message });
                }
            }
            return 0;
        }

        private static object Handle(string op, JsonElement root)
        {
            switch (op)
            {
                case "schema":
                    return new
                    {
                        ok = true, op,
                        nativeContract = ElderModel9Contract.NativeContractVersion,
                        actionSchema = ElderModel9Contract.ActionSchemaVersion,
                        observationSchema = ElderModel9Contract.ObservationSchemaVersion,
                        environmentSchema = ElderModel9Contract.EnvironmentSchemaVersion,
                        replaySchema = ElderModel9Contract.ReplaySchemaVersion,
                        coreWidth = ElderModel9Contract.CoreWidth,
                        coreFeatures = ElderModel9Encoder.CoreFeatureNames(),
                        sets = ElderModel9Contract.Sets.Select(s => new { name = s.Name, rows = s.Rows, width = s.Width }).ToArray(),
                        services = ElderModel9Contract.GoodsServices,
                        strategic = ElderModel9Contract.StrategicControls,
                        economyMix = ElderModel9Contract.EconomyMix,
                        movementSlots = ElderModel9Contract.MovementSlots,
                        nativeMovement = ElderModel9Contract.NativeMovement,
                        regionMovementSlots = ElderModel9Contract.RegionMovementSlots,
                        keepIntent = ElderMovement.KeepIntent, regionBase = ElderMovement.RegionBase,
                        regionFeatures = ElderModel9Encoder.RegionFeatureNames(),
                        buildHash = ElderModel9Environment.BuildHash
                    };
                case "reset":
                {
                    // withheld: SimulationConfig.TrainingWithheldTechnologies for capability fixtures (0 = the game).
                    int withheld = root.TryGetProperty("withheld", out var wh) ? wh.GetInt32() : 0;
                    byte[] snapshot = root.TryGetProperty("snapshot", out var s) && s.ValueKind == JsonValueKind.String
                        ? Convert.FromBase64String(s.GetString())
                        : ElderModel9Environment.GenerateOpening(root.GetProperty("seed").GetInt32(),
                            root.TryGetProperty("players", out var p) ? p.GetInt32() : 8, out _, 40, withheld);
                    bool movement = root.TryGetProperty("movement", out var m) && m.GetBoolean();
                    environment = new ElderModel9Environment { WithheldTechnologies = withheld };
                    var step = environment.Reset(snapshot, movement);
                    return new { ok = true, op, step = Encode(step), fingerprint = environment.Replay.InitialFingerprint,
                        configurationHash = environment.Replay.ConfigurationHash, buildHash = environment.Replay.BuildHash };
                }
                case "step":
                {
                    if (environment == null) throw new InvalidOperationException("reset required");
                    var actions = new Dictionary<int, ElderDirective>();
                    if (root.TryGetProperty("actions", out var a) && a.ValueKind == JsonValueKind.Object)
                        foreach (var item in a.EnumerateObject())
                            actions[int.Parse(item.Name)] = Directive(item.Value);
                    var step = environment.Step(actions);
                    return new { ok = true, op, step = Encode(step) };
                }
                case "verify_replay":
                    if (environment == null) throw new InvalidOperationException("reset required");
                    ElderModel9Environment.VerifyReplay(environment.Replay);
                    return new { ok = true, op, steps = environment.Replay.Steps.Count };
                case "capture":
                    if (environment == null) throw new InvalidOperationException("reset required");
                    return new { ok = true, op, snapshot = Convert.ToBase64String(SaveCodec.Save(environment.World)) };
                default:
                    throw new ArgumentException("Unknown model9 operation: " + op);
            }
        }

        private static ElderDirective Directive(JsonElement e)
        {
            var d = new ElderDirective { SchemaVersion = ElderModel9Contract.ActionSchemaVersion };
            double[] strategic = e.TryGetProperty("strategic", out var st) ? st.Deserialize<double[]>(Json) : null;
            if (strategic != null && (strategic.Length == 4 || strategic.Length == 5))
            {
                d.RetainedFieldOperation = strategic[0]; d.RetainedHerdFraction = strategic[1]; d.DevelopmentBudget = strategic[2]; d.FieldDevelopmentShare = strategic[3];
                if (strategic.Length == 5) d.FishingDevelopmentShare = strategic[4];
            }
            if (e.TryGetProperty("service", out var sv) && sv.ValueKind == JsonValueKind.Array) d.ServiceLevels = sv.Deserialize<double[]>(Json);
            d.ClothingRoute = e.TryGetProperty("clothingRoute", out var cr) ? cr.GetInt32() : -1;
            d.ShelterRoute = e.TryGetProperty("shelterRoute", out var sr) ? sr.GetInt32() : -1;
            d.MovementCandidate = e.TryGetProperty("movement", out var mv) ? mv.GetInt32() : ElderModel9Contract.NativeMovement;
            d.FissionPolicy = e.TryGetProperty("fission", out var fp) ? fp.GetInt32() : -1;
            // Contract 9: capital commitment span (0 hold, 1 this turn only, 2-6 commit); absent is 1.
            d.CommitTurns = e.TryGetProperty("commit", out var ct) ? ct.GetInt32() : 1;
            return d;
        }

        private static object Encode(ElderModel9Step step) => new
        {
            turn = step.Turn, years = step.Years, fingerprint = step.Fingerprint, nativeMs = step.NativeMilliseconds,
            villages = step.Villages.Select(v => new
            {
                camp = v.Camp, society = v.Society, parentSociety = v.ParentSociety, player = v.Player,
                priorPopulation = v.PriorPopulation, population = v.Population, qol = v.QoL, nutrition = v.Nutrition,
                clothing = v.Clothing, housing = v.Housing, years = v.Years, welfare = v.Welfare,
                terminated = v.Terminated, founded = v.Founded, interventions = v.Interventions.Length, privileged = v.Privileged,
                observation = v.Observation == null ? null : new
                {
                    core = v.Observation.Core, corePresent = v.Observation.CorePresent,
                    sets = v.Observation.Sets, rowPresent = v.Observation.RowPresent,
                    movementMask = v.Observation.MovementMask,
                    clothingRouteMask = v.Observation.ClothingRouteMask, shelterRouteMask = v.Observation.ShelterRouteMask,
                    economyMixMask = v.Observation.EconomyMixMask,
                    commitHoldMask = v.Observation.CommitmentHoldMask,
                    regionMovementMask = v.Observation.RegionMovementMask
                },
                teacher = v.Teacher == null ? null : new
                {
                    strategic = new[] { v.Teacher.Action.RetainedFieldOperation, v.Teacher.Action.RetainedHerdFraction,
                        v.Teacher.Action.DevelopmentBudget, v.Teacher.Action.FieldDevelopmentShare, v.Teacher.Action.FishingDevelopmentShare },
                    strategicMask = new[] { v.Teacher.StrategicFieldMask, v.Teacher.StrategicHerdMask, v.Teacher.DevelopmentMask, v.Teacher.DevelopmentShareMask, v.Teacher.FishingShareMask },
                    service = v.Teacher.Action.ServiceLevels, serviceMask = v.Teacher.ServiceMask,
                    clothingRoute = v.Teacher.Action.ClothingRoute, clothingRouteLabel = v.Teacher.ClothingRouteMask,
                    shelterRoute = v.Teacher.Action.ShelterRoute, shelterRouteLabel = v.Teacher.ShelterRouteMask,
                    movement = v.Teacher.Action.MovementCandidate, movementLabel = v.Teacher.MovementMask,
                    fission = v.Teacher.Action.FissionPolicy, repair = v.Teacher.RepairSize
                }
            }).ToArray()
        };

        private static void Write(object value)
        {
            Console.Out.WriteLine(JsonSerializer.Serialize(value, Json));
            Console.Out.Flush();
        }
    }
}
