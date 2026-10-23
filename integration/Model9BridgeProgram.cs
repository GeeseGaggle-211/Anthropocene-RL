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
        private static string BuildMetadata(string key) => typeof(Model9BridgeProgram).Assembly
            .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
            .Cast<System.Reflection.AssemblyMetadataAttribute>().FirstOrDefault(a => a.Key == key)?.Value ?? "unknown";
        private static readonly string BridgeSha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
            System.IO.File.ReadAllBytes(typeof(Model9BridgeProgram).Assembly.Location))).ToLowerInvariant();
        private static readonly JsonSerializerOptions Json = new JsonSerializerOptions
        {
            IncludeFields = true, PropertyNameCaseInsensitive = true, WriteIndented = false
        };
        private static readonly JsonSerializerOptions LaborJson = new JsonSerializerOptions
        {
            IncludeFields = true, PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = false
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
                        actorLabor = new
                        {
                            schemaVersion = ElderActorLaborContract.RegistryVersion,
                            nativeContract = ElderActorLaborContract.NativeContractVersion,
                            actionSchema = ElderActorLaborContract.ActionSchemaVersion,
                            observationSchema = ElderActorLaborContract.ObservationSchemaVersion,
                            environmentSchema = ElderActorLaborContract.EnvironmentSchemaVersion,
                            replaySchema = ElderActorLaborContract.ReplaySchemaVersion,
                            featureFlag = "actorLaborRetryV1",
                            specialties = ElderActorLaborContract.SpecialtyIds,
                            activities = ElderActorLaborContract.ActivityIds,
                            specialtyRows = ElderActorLaborContract.SpecialtyRows,
                            roleRows = ElderActorLaborContract.RoleRows,
                            activityRows = ElderActorLaborContract.ActivityRows,
                            resourceRows = ElderActorLaborContract.ResourceRows,
                            resources = ElderActorLaborContract.ResourceIds,
                            registryDigest = ElderActorLaborContract.RegistryDigest,
                            reasonRegistryVersion = ElderActorLaborContract.ReasonRegistryVersion,
                            reasonIds = ElderActorLaborContract.ReasonIds,
                            reasonRegistryDigest = ElderActorLaborContract.ReasonRegistryDigest,
                            populationAuthority = "full-decision-boundary-controllable-population",
                            apportionment = "largest-remainder-stable-ordinal",
                            activitySemantics = "staffing-work-share-not-output-promise",
                            maxProposals = ElderActorLaborContract.MaxProposals,
                            attemptFeasibilityCostMaximum = ElderActorLaborContract.FeasibilityCostMaximum,
                            decisionFeasibilityCostMaximum = ElderActorLaborContract.DecisionFeasibilityCostMaximum,
                            fallbackSurcharge = 0,
                            settlementGuardCostMaximum = 1
                        },
                        decisionDiagnosticsSchema = 1,
                        diagnosticEnvelopeSchema = 3,
                        economicDiagnosticsSchema = 1,
                        rawMaterialDiagnosticsSchema = 2,
                        bridgeBuildHash = typeof(Model9BridgeProgram).Module.ModuleVersionId.ToString("N"),
                        bridgeSha256 = BridgeSha256,
                        configurationHash = SaveCodec.ConfigurationHash(Config(root)),
                        players = Config(root).Players,
                        featureFlags = FeatureFlags(Config(root)),
                        producerFlags = new Dictionary<string, string> { ["sim_rules"] = Environment.GetEnvironmentVariable("ANTHROPOCENE_SIM_RULES") ?? "",
                            ["sim_statics"] = Environment.GetEnvironmentVariable("ANTHROPOCENE_SIM_STATICS") ?? "" },
                        simulatorCommit = BuildMetadata("AnthropoceneSimulatorCommit"),
                        simulatorSourceClean = BuildMetadata("AnthropoceneSimulatorClean") == "true",
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
                    int declaredSeed = root.GetProperty("seed").GetInt32();
                    RequireUnsealedSeed(declaredSeed);
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
                    // Measurement-only static switches (ANTHROPOCENE_SIM_STATICS, e.g. "NeolithicProduction.SiteTenureRepointDiagnosticOff=true").
                    foreach (var item in (Environment.GetEnvironmentVariable("ANTHROPOCENE_SIM_STATICS") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries))
                    {
                        var kv = item.Split('=', 2); int dot = kv[0].LastIndexOf('.');
                        var type = typeof(SimulationConfig).Assembly.GetType("Anthropocene.Simulation." + kv[0].Substring(0, dot).Trim());
                        var field = type?.GetField(kv[0].Substring(dot + 1).Trim(), System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                        if (field == null) throw new InvalidOperationException("Unknown simulator static " + kv[0]);
                        field.SetValue(null, Convert.ChangeType(kv[1].Trim(), field.FieldType, System.Globalization.CultureInfo.InvariantCulture));
                    }
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
                            : ElderModel9Environment.GenerateOpening(declaredSeed,
                                root.TryGetProperty("players", out var p) ? p.GetInt32() : 8, out _, 40, withheld);
                    }
                    finally { ElderModel9Environment.ConfigOverride = previous; }
                    var snapshotWorld = SaveCodec.Load(snapshot);
                    RequireUnsealedSeed(snapshotWorld.Seed);
                    if (snapshotWorld.Seed != declaredSeed) throw new InvalidOperationException("Snapshot root differs from declared seed.");
                    bool movement = root.TryGetProperty("movement", out var m) && m.GetBoolean();
                    environment = new ElderModel9Environment { WithheldTechnologies = withheld,
                        EconomicDiagnosticsEnabled = root.TryGetProperty("economicDiagnostics", out var economic) && economic.GetBoolean(),
                        ActorLaborEnabled = root.TryGetProperty("actorLabor", out var actorLabor) && actorLabor.GetBoolean() };
                    forks.Clear();
                    var step = environment.Reset(snapshot, movement);
                    return new { ok = true, op, step = Encode(step), fingerprint = environment.Replay.InitialFingerprint,
                        actorLaborEnabled = environment.ActorLaborEnabled,
                        rawMaterialsEnabled = environment.World.Config.ElderRawMaterialsEnabled,
                        sourceSeed = environment.World.Seed,
                        configurationHash = environment.Replay.ConfigurationHash, buildHash = environment.Replay.BuildHash };
                }
                case "validate_snapshot":
                {
                    var bytes = Convert.FromBase64String(root.GetProperty("snapshot").GetString());
                    var world = SaveCodec.Load(bytes);
                    RequireUnsealedSeed(world.Seed);
                    return new { ok = true, op, sourceSeed = world.Seed, turn = world.Turn,
                        fingerprint = SaveCodec.Fingerprint(world), payloadFingerprint = PayloadFingerprint(bytes),
                        rawMaterialsEnabled = world.Config.ElderRawMaterialsEnabled, featureFlags = FeatureFlags(world.Config),
                        configurationHash = SaveCodec.ConfigurationHash(world.Config), players = world.Config.Players,
                        livePlayerCount = world.PlayerCount };
                }
                case "reset_metadata":
                {
                    int seed = root.GetProperty("seed").GetInt32(); RequireUnsealedSeed(seed);
                    if (root.TryGetProperty("snapshot", out var snap))
                    {
                        var world = SaveCodec.Load(Convert.FromBase64String(snap.GetString()));
                        RequireUnsealedSeed(world.Seed);
                        if (world.Seed != seed) throw new InvalidOperationException("Snapshot root differs from declared seed.");
                        return new { ok = true, op, sourceSeed = world.Seed, turn = (int?)world.Turn,
                            rawMaterialsEnabled = world.Config.ElderRawMaterialsEnabled, featureFlags = FeatureFlags(world.Config),
                            configurationHash = SaveCodec.ConfigurationHash(world.Config), players = world.Config.Players,
                            livePlayerCount = world.PlayerCount };
                    }
                    var config = Config(root); config.Validate();
                    return new { ok = true, op, sourceSeed = seed, turn = (int?)null,
                        rawMaterialsEnabled = config.ElderRawMaterialsEnabled, featureFlags = FeatureFlags(config),
                        configurationHash = SaveCodec.ConfigurationHash(config), players = config.Players };
                }
                case "validate_fingerprint_snapshot":
                {
                    byte[] bytes = Convert.FromBase64String(root.GetProperty("snapshot").GetString());
                    var world = ValidateFingerprintSnapshot(bytes);
                    return new { ok = true, op, sourceSeed = world.Seed, turn = world.Turn,
                        payloadFingerprint = PayloadFingerprint(bytes) };
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
                case "validate_labor":
                {
                    if (environment == null) throw new InvalidOperationException("reset required");
                    int attempt = root.TryGetProperty("attempt", out var at) ? at.GetInt32() : 0;
                    var actions = ParseActions(root.GetProperty("actions"), actorLabor: true);
                    var digests = ParseDigests(root);
                    var attempts = environment.ValidateLabor(actions, attempt, digests);
                    return new { ok = true, op, attempts = attempts.OrderBy(p => p.Key)
                        .ToDictionary(p => p.Key, p => LaborElement(p.Value)) };
                }
                case "step_labor":
                {
                    if (environment == null) throw new InvalidOperationException("reset required");
                    var actions = ParseActions(root.GetProperty("actions"), actorLabor: true);
                    var retries = new Dictionary<int, ElderDirective>();
                    if (root.TryGetProperty("retryActions", out var retryActions) &&
                        retryActions.ValueKind == JsonValueKind.Object)
                        foreach (var item in retryActions.EnumerateObject())
                            retries[int.Parse(item.Name)] = LaborRetry(item.Value);
                    var step = environment.StepLabor(actions, retries, ParseDigests(root));
                    return new { ok = true, op, step = Encode(step),
                        laborFeedback = step.Villages.Where(v => v.LaborPrevious != null)
                            .OrderBy(v => v.Camp).ToDictionary(v => v.Camp,
                                v => LaborElement(v.LaborPrevious)) };
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
                case "fingerprint_snapshot":
                {
                    if (environment == null) throw new InvalidOperationException("reset required");
                    var w = environment.World;
                    string fingerprint = SaveCodec.Fingerprint(w);
                    // Exact canonical ordering/exclusions used by SaveCodec.Fingerprint. Export fails closed if
                    // that simulator contract changes: the full decompressed payload must hash to the native value.
                    var copy = new WorldState { MaterialSources = w.MaterialSources, MaterialKnowledge = w.MaterialKnowledge,
                        FixedHousing = w.FixedHousing?.OrderBy(s => s.CampId).ThenBy(s => s.Tile).ToList(),
                        FisherySites = w.FisherySites?.OrderBy(s => s.TileId).ToList(), DemographicMigrations = w.DemographicMigrations,
                        Version = w.Version, Seed = w.Seed, Turn = w.Turn, NextId = w.NextId, PlayerCount = w.PlayerCount,
                        Days = w.Days, Config = w.Config, Tiles = w.Tiles, Camps = w.Camps.OrderBy(c => c.Id).ToList(),
                        Societies = w.Societies.OrderBy(s => s.Id).ToList(), Improvements = w.Improvements.OrderBy(i => i.Id).ToList(),
                        Hunters = w.Hunters.OrderBy(h => h.Id).ToList(), Herds = w.Herds.OrderBy(h => h.Id).ToList(),
                        Rivers = w.Rivers, History = w.History, Generation = null, Resources = w.Resources,
                        AutonomousReceipts = w.AutonomousReceipts };
                    byte[] snapshot = SaveCodec.Save(copy);
                    if (PayloadFingerprint(snapshot) != fingerprint || SaveCodec.Fingerprint(w) != fingerprint)
                        throw new InvalidOperationException("Canonical fingerprint payload mismatch or export mutated live state.");
                    return new { ok = true, op, snapshot = Convert.ToBase64String(snapshot), fingerprint,
                        sourceSeed = w.Seed, turn = w.Turn, fingerprintContract = "savecodec-canonical-writepayload-sha256-v1" };
                }
                default:
                    throw new ArgumentException("Unknown model9 operation: " + op);
            }
        }

        private static SimulationConfig Config(JsonElement root)
        {
            var config = SimulationConfig.Small(root.TryGetProperty("players", out var p) ? p.GetInt32() : 8);
            foreach (var item in (Environment.GetEnvironmentVariable("ANTHROPOCENE_SIM_RULES") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                var kv = item.Split('=', 2); object target = config;
                var field = typeof(SimulationConfig).GetField(kv[0].Trim());
                if (field == null) { target = config.Neolithic; field = typeof(NeolithicConfig).GetField(kv[0].Trim()); }
                if (field == null) throw new InvalidOperationException("Unknown simulator rule " + kv[0]);
                field.SetValue(target, Convert.ChangeType(kv[1].Trim(), field.FieldType, System.Globalization.CultureInfo.InvariantCulture));
            }
            if (root.TryGetProperty("width", out var width)) config.Width = width.GetInt32();
            if (root.TryGetProperty("height", out var height)) config.Height = height.GetInt32();
            if (root.TryGetProperty("separation", out var separation)) config.StartSeparation = separation.GetInt32();
            return config;
        }

        private static Dictionary<string, bool> FeatureFlags(SimulationConfig config)
        {
            var flags = new Dictionary<string, bool>();
            foreach (var field in typeof(SimulationConfig).GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public))
                if (field.FieldType == typeof(bool)) flags["SimulationConfig." + field.Name] = (bool)field.GetValue(config);
            foreach (var field in typeof(NeolithicConfig).GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public))
                if (field.FieldType == typeof(bool)) flags["NeolithicConfig." + field.Name] = (bool)field.GetValue(config.Neolithic);
            return flags;
        }

        private static string PayloadFingerprint(byte[] snapshot)
        {
            using var input = new System.IO.MemoryStream(snapshot);
            using var zip = new System.IO.Compression.GZipStream(input, System.IO.Compression.CompressionMode.Decompress);
            using var hash = System.Security.Cryptography.SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(zip)).Replace("-", "");
        }

        private static byte[] Header(WorldState world)
        {
            using var output = new System.IO.MemoryStream();
            var serializer = new System.Runtime.Serialization.DataContractSerializer(typeof(WorldState),
                new System.Runtime.Serialization.DataContractSerializerSettings { MaxItemsInObjectGraph = int.MaxValue });
            using (var xml = System.Xml.XmlDictionaryWriter.CreateBinaryWriter(output, null, null, false)) serializer.WriteObject(xml, world);
            return output.ToArray();
        }

        private static WorldState ValidateFingerprintSnapshot(byte[] snapshot)
        {
            using var input = new System.IO.MemoryStream(snapshot);
            using var zip = new System.IO.Compression.GZipStream(input, System.IO.Compression.CompressionMode.Decompress);
            using var reader = new System.IO.BinaryReader(zip);
            string magic = reader.ReadString();
            if (!new[] { "ANTHROPOCENE-11", "ANTHROPOCENE-12", "ANTHROPOCENE-13", "ANTHROPOCENE-14", "ANTHROPOCENE-15" }.Contains(magic))
                throw new InvalidOperationException("Unknown fingerprint-save envelope.");
            bool hasFlags = magic == "ANTHROPOCENE-14" || magic == "ANTHROPOCENE-15";
            byte flags = hasFlags ? reader.ReadByte() : (byte)0;
            int size = reader.ReadInt32();
            if (size < 1 || size > 512 * 1024 * 1024) throw new InvalidOperationException("Invalid fingerprint-save header size.");
            byte[] header = reader.ReadBytes(size);
            var serializer = new System.Runtime.Serialization.DataContractSerializer(typeof(WorldState),
                new System.Runtime.Serialization.DataContractSerializerSettings { MaxItemsInObjectGraph = int.MaxValue });
            WorldState world;
            using (var xml = System.Xml.XmlDictionaryReader.CreateBinaryReader(header, System.Xml.XmlDictionaryReaderQuotas.Max))
                world = (WorldState)serializer.ReadObject(xml);
            RequireUnsealedSeed(world.Seed);
            if (world.Generation != null || !Header(world).SequenceEqual(header))
                throw new InvalidOperationException("Fingerprint-save header is not the complete canonical serialized header.");
            // Fingerprint deliberately omits nonphysical Generation, while ordinary SaveCodec.Load requires it.
            // Supply only that validator metadata in a temporary copy; never rewrite the compared artifact/tail.
            world.Generation = new GenerationReport();
            byte[] validationHeader = Header(world);
            using var output = new System.IO.MemoryStream();
            using (var compressed = new System.IO.Compression.GZipStream(output, System.IO.Compression.CompressionLevel.Fastest, true))
            using (var writer = new System.IO.BinaryWriter(compressed))
            {
                writer.Write(magic); if (hasFlags) writer.Write(flags);
                writer.Write(validationHeader.Length); writer.Write(validationHeader);
                zip.CopyTo(compressed);
            }
            return SaveCodec.Load(output.ToArray());
        }

        private static void RequireUnsealedSeed(int seed)
        {
            if (seed >= 3000 && seed <= 3059) throw new InvalidOperationException("Confirmation seed is sealed.");
        }

        private static JsonElement LaborElement(object value)
            => value == null ? JsonSerializer.SerializeToElement<object>(null,
                LaborJson) : JsonSerializer.SerializeToElement(value,
                    value.GetType(), LaborJson);

        private static Dictionary<int, ElderDirective> ParseActions(
            JsonElement value, bool actorLabor)
        {
            if (value.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("actions must be an object");
            var result = new Dictionary<int, ElderDirective>();
            foreach (var item in value.EnumerateObject())
                result[int.Parse(item.Name)] = actorLabor
                    ? ActorLaborDirective(item.Value) : Directive(item.Value);
            return result;
        }

        private static Dictionary<int, string> ParseDigests(JsonElement root)
        {
            if (!root.TryGetProperty("catalogDigests", out var value) ||
                value.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("catalogDigests must bind every labor action");
            var result = new Dictionary<int, string>();
            foreach (var item in value.EnumerateObject())
                result[int.Parse(item.Name)] = item.Value.GetString();
            return result;
        }

        private static ElderDirective ActorLaborDirective(JsonElement e)
        {
            string[] exact = { "schemaVersion", "movement", "fission",
                "tradeValue", "tradeGate", "tradeShare",
                "laborSpecialtyAllocation", "laborActivityAllocation" };
            var names = new HashSet<string>(e.EnumerateObject().Select(p => p.Name),
                StringComparer.Ordinal);
            if (!names.SetEquals(exact))
                throw new ArgumentException("schema-12 action fields must be exactly: " +
                    string.Join(",", exact));
            if (e.GetProperty("schemaVersion").GetInt32() !=
                ElderActorLaborContract.ActionSchemaVersion)
                throw new ArgumentException("unsupported actor-labor action schema");
            var d = new ElderDirective
            {
                SchemaVersion = ElderActorLaborContract.ActionSchemaVersion,
                MovementCandidate = e.GetProperty("movement").GetInt32(),
                FissionPolicy = e.GetProperty("fission").GetInt32(),
                TradeValue = e.GetProperty("tradeValue").Deserialize<double[]>(Json),
                TradeGate = e.GetProperty("tradeGate").Deserialize<int[]>(Json),
                TradeExportShare = e.GetProperty("tradeShare").GetDouble(),
                LaborSpecialtyAllocation = e.GetProperty(
                    "laborSpecialtyAllocation").Deserialize<double[]>(Json),
                LaborActivityAllocation = e.GetProperty(
                    "laborActivityAllocation").Deserialize<double[][]>(Json)
            };
            return d;
        }

        private static ElderDirective LaborRetry(JsonElement e)
        {
            string[] exact = { "laborSpecialtyAllocation",
                "laborActivityAllocation" };
            var names = new HashSet<string>(e.EnumerateObject().Select(p => p.Name),
                StringComparer.Ordinal);
            if (!names.SetEquals(exact))
                throw new ArgumentException("labor retry fields must be exactly: " +
                    string.Join(",", exact));
            return new ElderDirective
            {
                SchemaVersion = ElderActorLaborContract.ActionSchemaVersion,
                LaborSpecialtyAllocation = e.GetProperty(
                    "laborSpecialtyAllocation").Deserialize<double[]>(Json),
                LaborActivityAllocation = e.GetProperty(
                    "laborActivityAllocation").Deserialize<double[][]>(Json)
            };
        }

        private static ElderDirective Directive(JsonElement e)
        {
            if (e.TryGetProperty("schemaVersion", out var actionSchema) &&
                actionSchema.GetInt32() == ElderActorLaborContract.ActionSchemaVersion)
                return ActorLaborDirective(e);
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

        private static object Encode(ElderModel9Step step)
        {
            var ordinary = new
            {
            turn = step.Turn, years = step.Years, fingerprint = step.Fingerprint, nativeMs = step.NativeMilliseconds,
            diagnosticSchemaVersion = step.DiagnosticSchemaVersion,
            // Explicit envelope zero means no preceding decision, including an immediate reset fork.
            // Do not infer absence from generic head records: envelope two may carry raw-only diagnostics.
            trajectoryId = step.DiagnosticSchemaVersion == 0 ? null : step.TrajectoryId,
            rawMaterialDiagnostics = step.RawMaterialDiagnostics.OrderBy(p => p.Key, StringComparer.Ordinal)
                .ToDictionary(p => p.Key, p => new
                {
                    schemaVersion = p.Value.SchemaVersion, campId = p.Value.CampId, decisionTurn = p.Value.DecisionTurn,
                    material = (int)p.Value.Material, requestId = p.Value.RequestId,
                    requestedUnits = p.Value.RequestedUnits, acceptedUnits = p.Value.AcceptedUnits,
                    realisedUnits = p.Value.RealisedUnits
                }),
            villages = step.Villages.Select(v => new
            {
                camp = v.Camp, society = v.Society, parentSociety = v.ParentSociety, player = v.Player,
                priorPopulation = v.PriorPopulation, population = v.Population, qol = v.QoL, nutrition = v.Nutrition,
                clothing = v.Clothing, housing = v.Housing, years = v.Years, welfare = v.Welfare,
                terminated = v.Terminated, founded = v.Founded, interventions = v.Interventions.Length, privileged = v.Privileged,
                tradeUtility = v.TradeUtility, specialistShare = v.SpecialistShare, goodsOutput = v.GoodsOutput,
                decisionDiagnostics = v.DecisionDiagnostics,
                observation = EncodeObservation(v),
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
            if (environment?.EconomicDiagnosticsEnabled != true) return ordinary;
            var encoded = JsonSerializer.SerializeToElement(ordinary, Json).EnumerateObject().ToDictionary(p => p.Name, p => (object)p.Value);
            encoded["restoredObservation"] = step.RestoredObservation;
            encoded["economicDiagnostics"] = step.EconomicDiagnostics.OrderBy(p => p.Key).ToDictionary(p => p.Key, p => p.Value);
            return encoded;
        }

        private static object EncodeObservation(ElderModel9Village village)
        {
            if (village?.Observation == null) return null;
            var observation = village.Observation;
            var ordinary = new
            {
                core = observation.Core, corePresent = observation.CorePresent,
                sets = observation.Sets, rowPresent = observation.RowPresent,
                movementMask = observation.MovementMask,
                clothingRouteMask = observation.ClothingRouteMask,
                shelterRouteMask = observation.ShelterRouteMask,
                economyMixMask = observation.EconomyMixMask,
                commitHoldMask = observation.CommitmentHoldMask,
                regionMovementMask = observation.RegionMovementMask
            };
            if (environment?.ActorLaborEnabled != true) return ordinary;
            if (village.LaborCatalog == null)
                throw new InvalidOperationException(
                    "Actor-labor observation lacks its origin catalog.");
            var encoded = JsonSerializer.SerializeToElement(ordinary, Json)
                .EnumerateObject().ToDictionary(p => p.Name,
                    p => (object)p.Value);
            encoded["laborCatalog"] = LaborElement(village.LaborCatalog);
            encoded["laborPrevious"] = LaborElement(village.LaborPrevious);
            return encoded;
        }

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
