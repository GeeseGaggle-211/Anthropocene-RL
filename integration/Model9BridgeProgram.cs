using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Anthropocene.Simulation;

// JSON-lines bridge for native Elder contract 8 (model9). One environment per process; the learner never
// reimplements physics. Ops: schema, reset {seed, players, movement} | {snapshot}, step {actions}, verify_replay,
// capture, fork, restore {fork}, drop {fork}, close. Missing actions run the native heuristic for that village.
namespace Anthropocene.RLBridge
{
    internal static class Model9BridgeProgram
    {
        private static readonly JsonSerializerOptions Json = new JsonSerializerOptions
        {
            IncludeFields = true, PropertyNameCaseInsensitive = true, WriteIndented = false
        };
        private static ElderModel9Environment environment;
        // Search forks of the current environment (op fork / restore / drop), cleared on reset.
        private static readonly Dictionary<int, ElderModel9Fork> forks = new Dictionary<int, ElderModel9Fork>();
        private static int nextFork = 1;

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
                        tradeGroups = ElderTrade.Groups, tradeFeatures = ElderModel9Encoder.TradeFeatureNames(),
                        tradeValueRange = new[] { ElderTrade.MinValue, ElderTrade.MaxValue },
                        activityFeatures = ElderModel9Encoder.ActivityFeatureNames(),
                        buildHash = ElderModel9Environment.BuildHash
                    };
                case "reset":
                {
                    // withheld: SimulationConfig.TrainingWithheldTechnologies for capability fixtures (0 = the game).
                    int withheld = root.TryGetProperty("withheld", out var wh) ? wh.GetInt32() : 0;
                    // Optional world size (scale probes): width, height, separation; absent keeps SimulationConfig.Small.
                    int width = root.TryGetProperty("width", out var wd) ? wd.GetInt32() : 0;
                    int height = root.TryGetProperty("height", out var ht) ? ht.GetInt32() : 0;
                    int separation = root.TryGetProperty("separation", out var sp) ? sp.GetInt32() : 0;
                    var previous = ElderModel9Environment.ConfigOverride;
                    // Optional simulator-rule override for paired re-measurement (ANTHROPOCENE_SIM_RULES, e.g.
                    // "SettlingGraceOnce=false;PlantTentsEnabled=true"): public SimulationConfig/NeolithicConfig fields by
                    // name, set by reflection so the bridge still builds against simulators without them.
                    string rules = Environment.GetEnvironmentVariable("ANTHROPOCENE_SIM_RULES");
                    if (width > 0 || height > 0 || separation > 0 || !string.IsNullOrWhiteSpace(rules))
                        ElderModel9Environment.ConfigOverride = c =>
                        {
                            if (width > 0) c.Width = width;
                            if (height > 0) c.Height = height;
                            if (separation > 0) c.StartSeparation = separation;
                            foreach (var item in (rules ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries))
                            {
                                var kv = item.Split('=', 2);
                                object target = c; var field = typeof(SimulationConfig).GetField(kv[0].Trim());
                                if (field == null) { target = c.Neolithic; field = typeof(NeolithicConfig).GetField(kv[0].Trim()); }
                                if (field == null) throw new InvalidOperationException("Unknown simulator rule " + kv[0]);
                                field.SetValue(target, Convert.ChangeType(kv[1].Trim(), field.FieldType, System.Globalization.CultureInfo.InvariantCulture));
                            }
                        };
                    byte[] snapshot;
                    try
                    {
                        snapshot = root.TryGetProperty("snapshot", out var s) && s.ValueKind == JsonValueKind.String
                            ? Convert.FromBase64String(s.GetString())
                            : ElderModel9Environment.GenerateOpening(root.GetProperty("seed").GetInt32(),
                                root.TryGetProperty("players", out var p) ? p.GetInt32() : 8, out _, 40, withheld);
                    }
                    finally { ElderModel9Environment.ConfigOverride = previous; }
                    bool movement = root.TryGetProperty("movement", out var m) && m.GetBoolean();
                    environment = new ElderModel9Environment { WithheldTechnologies = withheld };
                    forks.Clear();
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
                case "fork":
                {
                    // Search branch point: returns its id and the decision step; every restore replays from here exactly.
                    if (environment == null) throw new InvalidOperationException("reset required");
                    int id = nextFork++;
                    forks[id] = environment.Fork(out var step);
                    return new { ok = true, op, fork = id, step = Encode(step) };
                }
                case "restore":
                {
                    if (environment == null) throw new InvalidOperationException("reset required");
                    int id = root.GetProperty("fork").GetInt32();
                    if (!forks.TryGetValue(id, out var fork)) throw new ArgumentException("unknown fork " + id);
                    return new { ok = true, op, step = Encode(environment.Restore(fork)) };
                }
                case "drop":
                    forks.Remove(root.GetProperty("fork").GetInt32());
                    return new { ok = true, op };
                case "verify_replay":
                    if (environment == null) throw new InvalidOperationException("reset required");
                    if (environment.Forked) throw new InvalidOperationException("replay verification is unavailable after a search fork");
                    ElderModel9Environment.VerifyReplay(environment.Replay);
                    return new { ok = true, op, steps = environment.Replay.Steps.Count };
                case "inspect":
                {
                    // Audit-only, read-only (docs/MODEL9_MOVEMENT_AUDIT.md): per-camp location, lifestyle, capital, last
                    // turn's food by source and the harvest-footprint stock via the pure LazyEcology.ProjectEcology read.
                    // Never MaterializeEcology; the fingerprint is taken before and after and must match.
                    if (environment == null) throw new InvalidOperationException("reset required");
                    string before = SaveCodec.Fingerprint(environment.World);
                    int[] extra = root.TryGetProperty("tiles", out var tl) && tl.ValueKind == JsonValueKind.Array ? tl.Deserialize<int[]>(Json) : new int[0];
                    var report = Inspect(environment.World, extra);
                    string after = SaveCodec.Fingerprint(environment.World);
                    if (before != after) throw new InvalidOperationException("inspect changed the world fingerprint");
                    return new { ok = true, op, fingerprint = after, inspect = report };
                }
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
            // Contract 11 trade posture (absent: native).
            if (e.TryGetProperty("tradeValue", out var tv) && tv.ValueKind == JsonValueKind.Array) d.TradeValue = tv.Deserialize<double[]>(Json);
            if (e.TryGetProperty("tradeGate", out var tg) && tg.ValueKind == JsonValueKind.Array) d.TradeGate = tg.Deserialize<int[]>(Json);
            d.TradeExportShare = e.TryGetProperty("tradeShare", out var ts) && ts.ValueKind == JsonValueKind.Number ? ts.GetDouble() : -1;
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
                tradeUtility = v.TradeUtility, specialistShare = v.SpecialistShare, goodsOutput = v.GoodsOutput,
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
                    fission = v.Teacher.Action.FissionPolicy, repair = v.Teacher.RepairSize,
                    tradeValue = v.Teacher.Action.TradeValue, tradeGate = v.Teacher.Action.TradeGate,
                    tradeShare = v.Teacher.Action.TradeExportShare, tradeLabel = v.Teacher.TradeMask
                }
            }).ToArray()
        };

        // Footprint stock of a camp standing on `tile`: accessible human food now (pure projection) and its
        // season-mean capacity, separately for land and aquatic tiles of the gather radius.
        private static object Footprint(WorldState w, int tile)
        {
            double land = 0, landCap = 0, aqua = 0, aquaCap = 0; int shallows = 0, landTiles = 0;
            foreach (int n in w.Map.Radius(tile, w.Config.GatherRadius))
            {
                var t = w.Tiles[n];
                if (t.Terrain == Terrain.Ocean || t.ElevationLevel >= 10) continue;
                var view = LazyEcology.ProjectEcology(w, n, w.Days);
                double cap = 0;
                for (int s = 0; s < 4; s++) cap += new EcologyCapacity(w, t, view, s).Food / 4;
                if (t.TraversableLand) { land += Math.Max(0, view.Food); landCap += cap; landTiles++; }
                else if (t.Aquatic) { aqua += Math.Max(0, view.Food); aquaCap += cap; if (t.Environment == EnvironmentKind.Shallows) shallows++; }
            }
            return new { tile, land, landCap, aqua, aquaCap, shallows, landTiles };
        }

        private static object Inspect(WorldState w, int[] extra)
        {
            var n = w.Config.Neolithic;
            var last = w.History.Count > 0 ? w.History[w.History.Count - 1] : null;
            var demo = last?.Demographics?.GroupBy(d => d.Camp).ToDictionary(g => g.Key, g => g.Last()) ?? new Dictionary<int, DemographicReport>();
            var camps = w.Camps.Where(c => c.Population > 0).OrderBy(c => c.Id).Select(c =>
            {
                var shelter = NeolithicProduction.HousingCapacity(w, c, w.Days);
                double e = c.RudimentaryExpectation;
                double cloth = c.ClothPerCapita, house = c.HousingPerCapita;
                double gear = n.RudimentaryWeightClothing * (1 - e * (1 - cloth)) + n.RudimentaryWeightHousing * (1 - e * (1 - house));
                double gearSettled = n.RudimentaryWeightClothing * cloth + n.RudimentaryWeightHousing * house;
                double cohesion = NeolithicProduction.SocialCohesion(n, NeolithicProduction.HousingShare(n, shelter, c.Population), c.Population);
                var owned = (w.FixedHousing ?? new List<FixedHousingStock>()).Where(s => s.CampId == c.Id).Select(s => s.Tile).Distinct().OrderBy(t => t)
                    .Select(t => new { tile = t, units = NeolithicProduction.FixedHousingAt(w, c.Id, t, w.Days) }).ToArray();
                var fields = Enumerable.Range(0, w.Tiles.Length).Where(i => w.Tiles[i].CultivationOperator == c.Id).ToArray();
                double fieldCapital = fields.Sum(i => LazyEcology.ProjectEcology(w, i, w.Days).CultivationCapital);
                double localCapital = w.Map.Radius(c.Tile, 1).Sum(i => Math.Max(0, LazyEcology.ProjectEcology(w, i, w.Days).CultivationCapital));
                demo.TryGetValue(c.Id, out var d);
                var memory = c.ElderMemory;
                return new
                {
                    camp = c.Id, society = c.SocietyId, player = c.Player, tile = c.Tile, elevation = w.Tiles[c.Tile].ElevationLevel,
                    population = c.Population, isVillage = c.IsVillage, villageTurns = c.VillageTurns, moves = c.Moves,
                    stationaryTurns = c.StationaryTurns, stationaryYears = c.StationaryYears, expectation = e,
                    hasIntent = memory != null && memory.HasIntent, intentTile = memory != null && memory.HasIntent ? memory.IntentTile : -1,
                    cultivation = CultivationEconomy.Enabled(c), husbandry = CultivationEconomy.HusbandryEnabled(c),
                    qol = c.QoL, food = c.FoodNeedsMetRatio, water = c.WaterNeedsMetRatio, cloth, house, gear, gearSettled, cohesion,
                    fixedHere = NeolithicProduction.FixedHousingAt(w, c, w.Days), fixedOwned = owned,
                    lightUnits = c.Inventory?.LightHousingUnits ?? 0, shelterFixed = shelter.Fixed, shelterLight = shelter.Light,
                    fieldTiles = fields.Length, fieldCapital, localCapital,
                    stores = c.Food, pots = c.Inventory?.Pots ?? 0,
                    harvest = d == null ? null : new
                    {
                        start = d.StartingPopulation, produced = d.FoodProduced, terrestrial = d.TerrestrialFood, marine = d.MarineFood,
                        lake = d.LakeFood, hunt = d.HuntFood, fish = d.FishFood, forage = d.ForageFood, animal = d.AnimalFood,
                        slaughter = d.SlaughterFood, consumed = d.FoodConsumed, required = d.FoodRequired, split = d.SplitOff,
                        emigrated = d.Emigrated, immigrated = d.Immigrated, births = d.GrossBirths, deaths = d.GrossDeaths
                    },
                    footprint = Footprint(w, c.Tile),
                    neighbours = w.Map.Neighbors(c.Tile).OrderBy(t => t).Select(t => Footprint(w, t)).ToArray()
                };
            }).ToArray();
            var fissions = last?.Fissions?.Select(f => new { parent = f.ParentCamp, daughter = f.DaughterCamp, parentSociety = f.ParentSociety,
                daughterSociety = f.DaughterSociety, tile = f.Tile, people = f.Daughter, trigger = f.Trigger.ToString() }).ToArray();
            return new { turn = w.Turn, days = w.Days, camps, fissions, migrations = last?.Migrations?.Count ?? 0,
                tiles = extra.Distinct().OrderBy(t => t).Select(t => Footprint(w, t)).ToArray() };
        }

        private static void Write(object value)
        {
            Console.Out.WriteLine(JsonSerializer.Serialize(value, Json));
            Console.Out.Flush();
        }
    }
}
