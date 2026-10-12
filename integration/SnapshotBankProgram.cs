using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.SnapshotBank
{
    internal static class SnapshotBankProgram
    {
        private const string FrozenSimulatorSha =
            "5559961164d6f9da8e6d20c1dc563ecec3eb04b7";
        private static readonly JsonSerializerOptions Json =
            new JsonSerializerOptions
            {
                IncludeFields = true,
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                WriteIndented = true
            };

        private static int Main(string[] args)
        {
            try
            {
                CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
                CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
                if (args.Length != 5 || !string.Equals(args[0], "generate",
                        StringComparison.OrdinalIgnoreCase) ||
                    args[1] != "--config" || args[3] != "--output")
                {
                    Console.Error.WriteLine(
                        "Usage: generate --config PLAN.json --output DIRECTORY");
                    return 2;
                }
                Generate(Path.GetFullPath(args[2]),
                    Path.GetFullPath(args[4]));
                return 0;
            }
            catch (Exception error)
            {
                Console.Error.WriteLine(error);
                return 1;
            }
        }

        private static void Generate(string configPath, string output)
        {
            var plan = JsonSerializer.Deserialize<GenerationPlan>(
                File.ReadAllText(configPath), Json)
                ?? throw new InvalidDataException(
                    "Generation plan deserialized to null.");
            ValidatePlan(plan);
            Directory.CreateDirectory(output);
            string snapshotDirectory = Path.Combine(output, "snapshots");
            Directory.CreateDirectory(snapshotDirectory);
            ElderObservationSchema.Validate();
            string codeHash = Sha256(File.ReadAllBytes(
                Assembly.GetExecutingAssembly().Location));
            string schemaHash = ObservationSchemaHash();
            var result = new GeneratorResult
            {
                Status = "PASS",
                NativeSimulatorSha = FrozenSimulatorSha,
                NativeSimulatorVersion = NativeElderEnvironment.SimulatorVersion,
                NativeBuildHash = NativeElderEnvironment.BuildHash,
                ObservationSchemaHash = schemaHash,
                GeneratorCodeSha256 = codeHash,
                GeneratorVersion = plan.NativeGeneratorVersion,
                GeneratorConfigSha256 = plan.GeneratorConfigSha256,
                ObservationSchemaVersion = ElderRLContract.ObservationSchemaVersion,
                ActionSchemaVersion = ElderRLContract.ActionSchemaVersion,
                EnvironmentSchemaVersion = ElderRLContract.EnvironmentSchemaVersion,
                CriticSidecarSchemaVersion = ElderRLContract.CriticSidecarSchemaVersion,
                TensorFeatureCount = ElderRLContract.TensorFeatureCount
            };
            var factory = new SnapshotScenarioFactory(plan);
            var orderedTrajectories = plan.Trajectories.OrderBy(item =>
                item.TrajectoryId, StringComparer.Ordinal).ToArray();
            var handledEventGroups = new HashSet<string>(
                StringComparer.Ordinal);
            foreach (var trajectory in orderedTrajectories)
            {
                if (IsEventAlignedF15(plan, trajectory))
                {
                    string group = trajectory.CounterfactualGroupId;
                    if (!handledEventGroups.Add(group)) continue;
                    GenerateEventAlignedTradePair(plan,
                        orderedTrajectories.Where(item =>
                            item.CounterfactualGroupId == group).ToArray(),
                        factory, snapshotDirectory, result, codeHash,
                        schemaHash);
                    continue;
                }
                try
                {
                    GenerateTrajectory(plan, trajectory, factory,
                        snapshotDirectory, result, codeHash, schemaHash);
                }
                catch (Exception error)
                {
                    result.RejectedCandidates.Add(Reject(trajectory,
                        error));
                }
            }
            if (result.Candidates.Count == 0)
                result.Status = "FAIL";
            string resultPath = Path.Combine(output, "native-result.json");
            File.WriteAllText(resultPath,
                JsonSerializer.Serialize(result, Json) + Environment.NewLine);
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                result.Status,
                candidates = result.Candidates.Count,
                rejected = result.RejectedCandidates.Count,
                result_path = resultPath
            }, Json));
        }

        private static void GenerateTrajectory(GenerationPlan plan,
            TrajectoryPlan trajectory, SnapshotScenarioFactory factory,
            string snapshotDirectory, GeneratorResult output,
            string codeHash, string schemaHash)
        {
            ScenarioBuildResult built = factory.Build(trajectory);
            byte[] fixture = SaveCodec.Save(built.World);
            int maximumStep = trajectory.SnapshotSteps.Max();
            var environment = new NativeElderEnvironment();
            var reset = environment.Reset(Request(plan, fixture,
                built.ControlledCampId, Math.Max(1, maximumStep + 1),
                SaveCodec.ConfigurationHash(built.World.Config)));
            ElderObservation observation = reset.Observation;
            bool initialCultivation = observation.Mask.FieldDevelopment;
            for (int step = 0; step <= maximumStep; step++)
            {
                if (trajectory.SnapshotSteps.Contains(step))
                {
                    byte[] snapshot = environment.CaptureSnapshot();
                    bool technologyTransition = !initialCultivation &&
                        observation.Mask.FieldDevelopment;
                    var accepted = AcceptCandidate(plan, trajectory,
                        step, snapshot, built.ControlledCampId,
                        built.SiteSelectionEvidence, technologyTransition,
                        snapshotDirectory, codeHash, schemaHash,
                        out string configHash);
                    output.Candidates.Add(accepted);
                    RecordConfigurationHash(output, configHash);
                }
                if (step == maximumStep) break;
                var advanced = environment.Step(TrajectoryDirective(
                    observation, step));
                if (advanced.Terminated || advanced.Observation == null)
                    throw new InvalidOperationException(
                        "Controlled camp terminated before a requested timestamp.");
                observation = advanced.Observation;
            }
        }

        private static bool IsEventAlignedF15(GenerationPlan plan,
            TrajectoryPlan trajectory) =>
            plan.F15SharedCaptureHorizonSteps > 0 &&
            trajectory.ScenarioFamily == "F15" &&
            (trajectory.CausalVariant == "mild-import" ||
                trajectory.CausalVariant == "surplus-export");

        private static void GenerateEventAlignedTradePair(
            GenerationPlan plan, TrajectoryPlan[] siblings,
            SnapshotScenarioFactory factory, string snapshotDirectory,
            GeneratorResult output, string codeHash, string schemaHash)
        {
            var trace = new List<object>();
            var accepted = new List<CandidateEnvelope>();
            try
            {
                ValidateEventAlignedPair(siblings);
                var states = siblings.OrderBy(item => item.CausalVariant,
                        StringComparer.Ordinal)
                    .Select(item => EventAlignedState(plan, item, factory))
                    .ToArray();
                int selectedDecision = -1;
                byte[][] selectedSnapshots = null;
                for (int decision = 1;
                    decision <= plan.F15SharedCaptureHorizonSteps;
                    decision++)
                {
                    foreach (var state in states)
                    {
                        var advanced = state.Environment.Step(
                            TrajectoryDirective(state.Observation,
                                decision - 1));
                        if (advanced.Terminated || advanced.Observation == null)
                            throw new InvalidOperationException(
                                "Controlled camp terminated inside the shared F15 " +
                                "capture horizon.");
                        state.Observation = advanced.Observation;
                        state.Snapshot = state.Environment.CaptureSnapshot();
                        var world = SaveCodec.Load(state.Snapshot);
                        var camp = world.Camps.Single(item =>
                            item.Id == state.Built.ControlledCampId);
                        state.Measurement = Measure(world, camp,
                            state.Observation, false);
                        state.PhysicalYears = world.Days /
                            world.Config.DaysPerYear;
                    }
                    if (Math.Abs(states[0].PhysicalYears -
                            states[1].PhysicalYears) > 1e-9)
                        throw new InvalidOperationException(
                            "Event-aligned siblings drifted to unequal physical time.");
                    trace.Add(EventTraceRow(decision, states));
                    if (states.All(EventDirectionSatisfied))
                    {
                        selectedDecision = decision;
                        selectedSnapshots = states.Select(item =>
                            item.Snapshot).ToArray();
                        break;
                    }
                }
                if (selectedDecision < 0 || selectedSnapshots == null)
                    throw new InvalidOperationException(
                        "Requested stratum was not achieved: shared F15 delivered-" +
                        "direction event within the fixed capture horizon.");

                var metadata = EventCaptureMetadata(plan, trace,
                    selectedDecision);
                for (int index = 0; index < states.Length; index++)
                {
                    var state = states[index];
                    bool technologyTransition = !state.InitialCultivation &&
                        state.Observation.Mask.FieldDevelopment;
                    var candidate = AcceptCandidate(plan, state.Plan,
                        selectedDecision, selectedSnapshots[index],
                        state.Built.ControlledCampId,
                        state.Built.SiteSelectionEvidence,
                        technologyTransition, snapshotDirectory, codeHash,
                        schemaHash, out string configHash, metadata);
                    accepted.Add(candidate);
                    RecordConfigurationHash(output, configHash);
                }
                output.Candidates.AddRange(accepted);
            }
            catch (Exception error)
            {
                foreach (var candidate in accepted)
                {
                    string path = Path.Combine(snapshotDirectory,
                        Path.GetFileName(candidate.SnapshotFile));
                    if (File.Exists(path)) File.Delete(path);
                }
                var metadata = EventCaptureMetadata(plan, trace, null);
                foreach (var sibling in siblings)
                    output.RejectedCandidates.Add(Reject(sibling, error,
                        metadata));
            }
        }

        private static CandidateEnvelope AcceptCandidate(GenerationPlan plan,
            TrajectoryPlan trajectory, int decisionIndex, byte[] snapshot,
            int controlledCamp, string siteEvidence,
            bool technologyTransition, string snapshotDirectory,
            string codeHash, string schemaHash,
            out string configurationHash,
            Dictionary<string, object> selectionMetadata = null)
        {
            var loaded = SaveCodec.Load(snapshot);
            string fingerprint = SaveCodec.Fingerprint(loaded);
            byte[] canonical = SaveCodec.Save(loaded);
            var roundtrip = SaveCodec.Load(canonical);
            string roundtripFingerprint = SaveCodec.Fingerprint(roundtrip);
            if (!string.Equals(fingerprint, roundtripFingerprint,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "SaveCodec.Save -> Load -> Fingerprint mismatch.");
            var environment = new NativeElderEnvironment();
            string expectedConfigurationHash =
                SaveCodec.ConfigurationHash(roundtrip.Config);
            var reset = environment.Reset(Request(plan, canonical,
                controlledCamp, plan.CalibrationRolloutSteps,
                expectedConfigurationHash));
            configurationHash = reset.ConfigurationHash;
            ValidateSchemaAndMasks(reset, expectedConfigurationHash,
                fingerprint);
            var tensor = ElderObservationSchema.Encode(reset.Observation);
            ValidateFiniteTensor(tensor);
            var samples = new List<NativeObservationSample>
            {
                new NativeObservationSample
                {
                    Source = "reset",
                    Tensor = tensor
                }
            };
            ElderStepResult first = null;
            ElderObservation current = reset.Observation;
            for (int step = 0; step < plan.CalibrationRolloutSteps; step++)
            {
                var advanced = environment.Step(CalibrationDirective(
                    plan, current, step));
                if (first == null) first = advanced;
                if (!(advanced.ElapsedYears > 0) ||
                    double.IsNaN(advanced.ElapsedYears) ||
                    double.IsInfinity(advanced.ElapsedYears))
                    throw new InvalidOperationException(
                        "Native Step did not advance finite positive physical time.");
                if (advanced.Observation != null)
                {
                    var encoded = ElderObservationSchema.Encode(
                        advanced.Observation);
                    ValidateFiniteTensor(encoded);
                    samples.Add(new NativeObservationSample
                    {
                        Source = "bounded-masked-rollout",
                        Tensor = encoded
                    });
                }
                if (advanced.Terminated) break;
                current = advanced.Observation;
            }
            if (first == null)
                throw new InvalidOperationException(
                    "Accepted candidate did not execute a validation Step.");
            NativeElderEnvironment.VerifyReplay(environment.ReplayLog,
                new PhaseExecution(2, 4, true, 1));

            // Measurements and tags always come from the Reset observation and
            // canonical saved WorldState, never from requested bands.
            var state = SaveCodec.Load(canonical);
            var camp = state.Camps.Single(item => item.Id == controlledCamp);
            var measured = Measure(state, camp, reset.Observation,
                technologyTransition);
            ValidateRequestedState(trajectory, reset.Observation,
                measured.Tags, decisionIndex, technologyTransition);
            string contentHash = Sha256(canonical);
            string observationHash = Sha256(JsonSerializer.SerializeToUtf8Bytes(
                tensor, Json));
            string snapshotId = SnapshotId(trajectory, decisionIndex,
                fingerprint);
            string fileName = snapshotId + ".snapshot";
            File.WriteAllBytes(Path.Combine(snapshotDirectory, fileName),
                canonical);
            double memoryAge = MemoryAgeYears(state, camp);
            bool[] movementCandidates = Enumerable.Range(0, 8)
                .Select(index => index == 0 ?
                    reset.Observation.Mask.AllowsMovement(-1) :
                    reset.Observation.Mask.AllowsMovement(index - 1))
                .ToArray();
            var selection = SelectionMetadata(trajectory, siteEvidence,
                selectionMetadata);
            var record = new
            {
                snapshot_id = snapshotId,
                content_sha256 = contentHash,
                content_bytes = canonical.Length,
                blob_path = "",
                native_fingerprint = fingerprint,
                partition = trajectory.Partition,
                scenario_family = trajectory.ScenarioFamily,
                scenario_version = trajectory.ScenarioVersion,
                world_seed = trajectory.WorldSeed,
                trajectory_family = trajectory.TrajectoryFamily,
                lineage_id = trajectory.LineageId,
                counterfactual_group_id = trajectory.CounterfactualGroupId,
                source_controller = trajectory.SourceController,
                controlled = new
                {
                    camp_id = camp.Id,
                    society_id = camp.SocietyId,
                    player_id = camp.Player
                },
                physical_years = state.Days / state.Config.DaysPerYear,
                decision_index = decisionIndex,
                native_simulator_sha = FrozenSimulatorSha,
                native_simulator_version =
                    NativeElderEnvironment.SimulatorVersion,
                configuration_hash = reset.ConfigurationHash,
                native_build_hash = reset.BuildHash,
                schema = new
                {
                    observation_schema_version =
                        ElderRLContract.ObservationSchemaVersion,
                    action_schema_version =
                        ElderRLContract.ActionSchemaVersion,
                    environment_schema_version =
                        ElderRLContract.EnvironmentSchemaVersion,
                    critic_sidecar_schema_version =
                        ElderRLContract.CriticSidecarSchemaVersion,
                    tensor_feature_count =
                        ElderRLContract.TensorFeatureCount,
                    observation_schema_hash = schemaHash
                },
                actual_strata = new
                {
                    population = reset.Observation.Core.Population,
                    connected_consumers =
                        reset.Observation.Core.ConnectedConsumers,
                    next_interval_years =
                        reset.Observation.Core.NextIntervalYears,
                    food_reserve_years =
                        reset.Observation.Core.FoodReserveYears,
                    reserve_horizon_ratio =
                        reset.Observation.Core.FoodReserveYears /
                        reset.Observation.Core.NextIntervalYears,
                    field_q = reset.Observation.Fields.CurrentIntensity,
                    fishing_q = measured.FishingQ,
                    herd_animals = reset.Observation.Herd.Animals,
                    herd_zero_displacement_capacity =
                        reset.Observation.Herd.ZeroDisplacementCapacity,
                    herd_absolute_capacity =
                        reset.Observation.Herd.AbsoluteCapacity,
                    herd_water_capacity =
                        reset.Observation.Herd.WaterCapacity,
                    herd_zero_displacement_ratio = Ratio(
                        reset.Observation.Herd.Animals,
                        reset.Observation.Herd.ZeroDisplacementCapacity),
                    herd_absolute_ratio = Ratio(
                        reset.Observation.Herd.Animals,
                        reset.Observation.Herd.AbsoluteCapacity),
                    herd_water_ratio = Ratio(
                        reset.Observation.Herd.Animals,
                        reset.Observation.Herd.WaterCapacity),
                    fishing_ground_count =
                        reset.Observation.Fishing.Length,
                    current_qol = reset.Observation.Core.QoL,
                    food_needs_met = reset.Observation.Core.FoodNeedsMet,
                    water_needs_met = reset.Observation.Core.WaterNeedsMet,
                    clothing_coverage =
                        reset.Observation.Core.ClothingCoverage,
                    housing_coverage =
                        reset.Observation.Core.HousingCoverage,
                    retained_field_capital = measured.RetainedFieldCapital,
                    retained_field_operation =
                        reset.Observation.Fields.CurrentOperation,
                    maintenance_worker_years =
                        measured.MaintenanceWorkerYears,
                    hand_tools_per_capita = measured.HandToolsPerCapita,
                    farm_tools_per_capita = measured.FarmToolsPerCapita,
                    materials_per_capita = measured.MaterialsPerCapita,
                    delivered_trade_total = measured.DeliveredTradeTotal,
                    delivered_import_total = measured.DeliveredImportTotal,
                    delivered_export_total = measured.DeliveredExportTotal,
                    delivered_food_import = measured.DeliveredFoodImport,
                    delivered_food_export = measured.DeliveredFoodExport,
                    delivered_material_tool_import =
                        measured.DeliveredMaterialToolImport,
                    import_dependence = measured.ImportDependence,
                    signed_migration = measured.SignedMigration,
                    fission_candidate_present =
                        reset.Observation.Fission.CandidatePresent,
                    fission_recent_split = measured.RecentFission,
                    fission_pressure = reset.Observation.Fission.Pressure,
                    fission_exit_opportunity =
                        reset.Observation.Fission.ExitOpportunityPressure,
                    fission_daughter_population =
                        reset.Observation.Fission.RecentDaughterPopulation,
                    parent_population_relief =
                        measured.ParentPopulationRelief,
                    field_frontier = measured.FieldFrontier,
                    cultivation_available = measured.Cultivation,
                    husbandry_available = measured.Husbandry,
                    fishing_available =
                        reset.Observation.Fishing.Length > 0,
                    owned_domestic_herd =
                        reset.Observation.Herd.Animals > 0,
                    structural_masks = new
                    {
                        field_retention =
                            reset.Observation.Mask.FieldRetention,
                        herd_retention =
                            reset.Observation.Mask.HerdRetention,
                        field_development =
                            reset.Observation.Mask.FieldDevelopment,
                        herd_development =
                            reset.Observation.Mask.HerdDevelopment,
                        movement = reset.Observation.Mask.Movement,
                        movement_candidates = movementCandidates
                    },
                    tags = measured.Tags
                },
                institutional_memory = new
                {
                    age_years = memoryAge,
                    maturity_20y = reset.Observation.Memory.Maturity[0],
                    maturity_60y = reset.Observation.Memory.Maturity[1],
                    maturity_180y = reset.Observation.Memory.Maturity[2]
                },
                classification = trajectory.Classification,
                generator = new
                {
                    name = "Anthropocene.SnapshotBank",
                    version = plan.NativeGeneratorVersion,
                    code_sha256 = codeHash,
                    generation_seed = plan.GenerationSeed,
                    rng_algorithm = "native-keyed-Numbers.Hash",
                    rng_stream = "world-seed:" + trajectory.WorldSeed +
                        "/trajectory:" + trajectory.TrajectoryId
                },
                selection_metadata = selection,
                observation_sha256 = observationHash,
                calibration_observation_count = samples.Count,
                validation = new
                {
                    save_load_fingerprint = true,
                    native_reset = true,
                    legal_native_step = true,
                    replay_verified = true,
                    finite_positive_elapsed_years = true,
                    schema_configuration_hash_match = true,
                    structural_masks_valid = true,
                    measured_strata_match = true,
                    elapsed_years = first.ElapsedYears,
                    post_step_fingerprint = first.StateFingerprint
                }
            };
            return new CandidateEnvelope
            {
                Record = record,
                SnapshotFile = Path.Combine("snapshots", fileName),
                CalibrationObservations = samples.ToArray()
            };
        }

        private sealed class Measurement
        {
            public bool Cultivation, Husbandry;
            public double? FishingQ;
            public double RetainedFieldCapital, MaintenanceWorkerYears,
                DeliveredTradeTotal, DeliveredImportTotal,
                DeliveredExportTotal, DeliveredFoodImport,
                DeliveredFoodExport, DeliveredMaterialToolImport,
                ImportDependence, SignedMigration, ParentPopulationRelief;
            public double HandToolsPerCapita, FarmToolsPerCapita,
                MaterialsPerCapita;
            public bool RecentFission;
            public string FieldFrontier;
            public string[] Tags;
        }

        private sealed class EventAlignedTrajectoryState
        {
            public TrajectoryPlan Plan;
            public ScenarioBuildResult Built;
            public NativeElderEnvironment Environment;
            public ElderObservation Observation;
            public bool InitialCultivation;
            public byte[] Snapshot;
            public Measurement Measurement;
            public double PhysicalYears;
        }

        private static EventAlignedTrajectoryState EventAlignedState(
            GenerationPlan plan, TrajectoryPlan trajectory,
            SnapshotScenarioFactory factory)
        {
            ScenarioBuildResult built = factory.Build(trajectory);
            byte[] fixture = SaveCodec.Save(built.World);
            var environment = new NativeElderEnvironment();
            var reset = environment.Reset(Request(plan, fixture,
                built.ControlledCampId,
                Math.Max(1, plan.F15SharedCaptureHorizonSteps + 1),
                SaveCodec.ConfigurationHash(built.World.Config)));
            return new EventAlignedTrajectoryState
            {
                Plan = trajectory,
                Built = built,
                Environment = environment,
                Observation = reset.Observation,
                InitialCultivation = reset.Observation.Mask.FieldDevelopment
            };
        }

        private static bool EventDirectionSatisfied(
            EventAlignedTrajectoryState state)
        {
            const double tolerance = 1e-9;
            if (!(state.Measurement.DeliveredTradeTotal > tolerance))
                return false;
            return state.Plan.CausalVariant == "mild-import" ?
                state.Measurement.DeliveredFoodImport > tolerance :
                state.Measurement.DeliveredFoodExport > tolerance;
        }

        private static object EventTraceRow(int decision,
            EventAlignedTrajectoryState[] states) => new
        {
            decision_index = decision,
            physical_years = states[0].PhysicalYears,
            variants = states.ToDictionary(item => item.Plan.CausalVariant,
                item => (object)new
                {
                    delivered_trade_total =
                        item.Measurement.DeliveredTradeTotal,
                    delivered_food_import =
                        item.Measurement.DeliveredFoodImport,
                    delivered_food_export =
                        item.Measurement.DeliveredFoodExport,
                    direction_satisfied = EventDirectionSatisfied(item)
                }, StringComparer.Ordinal)
        };

        private static Dictionary<string, object> EventCaptureMetadata(
            GenerationPlan plan, List<object> trace, int? selectedDecision)
            => new Dictionary<string, object>
            {
                ["shared_capture_rule"] =
                    "shared-f15-delivered-direction-v1",
                ["shared_capture_horizon_steps"] =
                    plan.F15SharedCaptureHorizonSteps,
                ["shared_capture_status"] = selectedDecision.HasValue ?
                    "captured" : "not-achieved",
                ["shared_capture_decision_index"] = selectedDecision,
                ["shared_capture_trace"] = trace
            };

        private static void ValidateEventAlignedPair(
            TrajectoryPlan[] siblings)
        {
            if (siblings.Length != 2)
                throw new InvalidDataException(
                    "Event-aligned F15 group must contain exactly two siblings.");
            var variants = siblings.Select(item => item.CausalVariant)
                .OrderBy(item => item, StringComparer.Ordinal).ToArray();
            if (!variants.SequenceEqual(new[]
                    { "mild-import", "surplus-export" }))
                throw new InvalidDataException(
                    "Event-aligned F15 group must pair mild-import and " +
                    "surplus-export.");
            var left = siblings[0];
            var right = siblings[1];
            bool matched = left.ScenarioFamily == "F15" &&
                right.ScenarioFamily == "F15" &&
                left.ScenarioVersion == right.ScenarioVersion &&
                left.WorldSeed == right.WorldSeed &&
                left.TrajectoryFamily == right.TrajectoryFamily &&
                left.LineageId == right.LineageId &&
                left.CounterfactualGroupId == right.CounterfactualGroupId &&
                left.SourceController == right.SourceController &&
                left.Classification == right.Classification &&
                left.Partition == right.Partition &&
                left.HistoricalOrConsumedDiagnostic ==
                    right.HistoricalOrConsumedDiagnostic &&
                left.SnapshotSteps.Length == 1 &&
                right.SnapshotSteps.Length == 1 &&
                left.Economy == right.Economy &&
                left.Population == right.Population &&
                left.FieldQ == right.FieldQ &&
                left.HerdSupportRatio == right.HerdSupportRatio &&
                left.ReserveIntervals == right.ReserveIntervals &&
                left.FishingGroundTarget == right.FishingGroundTarget &&
                left.SiteSupportQuantile == right.SiteSupportQuantile &&
                left.ToolAbundance == right.ToolAbundance &&
                left.AmenityCoverage == right.AmenityCoverage &&
                left.Frontier == right.Frontier;
            if (!matched)
                throw new InvalidDataException(
                    "Event-aligned F15 siblings must differ only in their " +
                    "declared causal variant and requested direction tag.");
        }

        private static Measurement Measure(WorldState state, Camp camp,
            ElderObservation observation, bool technologyTransition)
        {
            bool cultivation = CultivationEconomy.Enabled(camp);
            bool husbandry = CultivationEconomy.HusbandryEnabled(camp);
            double? fishingQ = observation.Fishing.Length == 0 ?
                (double?)null : observation.Fishing.Average(item =>
                    item.CurrentIntensity);
            string frontier = cultivation ? SettlementProductionFrontier
                .ResolveCultivation(state, camp).Identity : "none";
            double memoryAge = MemoryAgeYears(state, camp);
            var recent = state.History.LastOrDefault();
            double imports = 0, exports = 0, foodImports = 0,
                foodExports = 0, materialToolImports = 0;
            foreach (var trade in recent?.Trades ?? new List<TradeRecord>())
            {
                double wantDelivered = trade.WantQuantity *
                    Math.Max(0, Math.Min(1, trade.Delivered));
                double giveDelivered = trade.GiveQuantity *
                    Math.Max(0, Math.Min(1, trade.GiveDelivered));
                if (trade.Requester == camp.Id)
                {
                    imports += wantDelivered;
                    exports += giveDelivered;
                    if (trade.Want == TradeGood.Food)
                        foodImports += wantDelivered;
                    if (trade.Give == TradeGood.Food)
                        foodExports += giveDelivered;
                    if (MaterialOrTool(trade.Want))
                        materialToolImports += wantDelivered;
                }
                else if (trade.Responder == camp.Id)
                {
                    imports += giveDelivered;
                    exports += wantDelivered;
                    if (trade.Give == TradeGood.Food)
                        foodImports += giveDelivered;
                    if (trade.Want == TradeGood.Food)
                        foodExports += wantDelivered;
                    if (MaterialOrTool(trade.Give))
                        materialToolImports += giveDelivered;
                }
            }
            double migration = 0;
            foreach (var flow in recent?.Migrations ??
                new List<MigrationReport>())
            {
                if (flow.DestinationCamp == camp.Id)
                    migration += flow.Migrants;
                if (flow.SourceCamp == camp.Id)
                    migration -= flow.Migrants;
            }
            var fission = (recent?.Fissions ?? new List<FissionReport>())
                .LastOrDefault(item => item.ParentCamp == camp.Id);
            double intervalNeed = Math.Max(1e-12,
                Math.Max(1, recent?.Demographics?.FirstOrDefault(item =>
                    item.Camp == camp.Id)?.StartingPopulation ??
                    camp.Population) * state.Config.Consumption *
                Math.Max(0, recent?.Duration ?? 0) / state.Config.BaseDays);
            var tags = new HashSet<string>(StringComparer.Ordinal);
            if (cultivation && !husbandry) tags.Add("cultivation-only");
            if (husbandry && !cultivation) tags.Add("husbandry-only");
            if (cultivation && husbandry)
                tags.Add("mixed-cultivation-husbandry");
            if (observation.Fishing.Length > 0)
            {
                tags.Add("fishing");
                if (!cultivation && !husbandry) tags.Add("fishing-only");
            }
            if (observation.Herd.Animals == 0) tags.Add("no-herd");
            if (memoryAge < 20) tags.Add("young-memory");
            if (memoryAge >= 180) tags.Add("mature-memory");
            double q = Math.Max(observation.Fields.CurrentIntensity,
                fishingQ ?? 0);
            if (q < .3) tags.Add("low-q");
            if (q >= .35 && q <= .45) tags.Add("near-knee-q");
            if (q >= .5) tags.Add("high-q");
            double? herdRatio = Ratio(observation.Herd.Animals,
                observation.Herd.ZeroDisplacementCapacity);
            if (herdRatio.HasValue && herdRatio.Value >= 1.3)
                tags.Add("overstocked-herd");
            if (observation.Core.FoodReserveYears /
                    observation.Core.NextIntervalYears >= 1 &&
                Math.Min(observation.Core.ClothingCoverage,
                    observation.Core.HousingCoverage) < .5)
                tags.Add("amenity-bottleneck");
            if (!observation.Mask.AllowsDevelopment())
                tags.Add("no-development");
            if (husbandry && observation.Herd.Animals == 0)
                tags.Add("knowledge-only-husbandry-no-herd");
            if (technologyTransition) tags.Add("technology-transition");
            if (imports + exports > 1e-9) tags.Add("trade-active");
            if (imports > exports + 1e-9) tags.Add("trade-importer");
            if (exports > imports + 1e-9) tags.Add("trade-exporter");
            if (foodImports > 1e-9) tags.Add("food-import");
            if (foodExports > 1e-9) tags.Add("food-export");
            if (materialToolImports > 1e-9)
                tags.Add("material-tool-import");
            if (foodImports / intervalNeed >= .25)
                tags.Add("trade-dependent");
            if (migration > 0)
            {
                tags.Add("migration-positive");
                tags.Add("immigration");
            }
            if (migration < 0)
            {
                tags.Add("migration-negative");
                tags.Add("emigration");
            }
            if (observation.Contacts.Length == 0 &&
                imports + exports <= 1e-9 && migration == 0)
                tags.Add("isolated");
            if (observation.Fission.CandidatePresent)
                tags.Add("fission-candidate");
            if (fission != null)
            {
                tags.Add("recent-fission");
                tags.Add("post-fission");
            }
            else if (observation.Fission.CandidatePresent)
                tags.Add("pre-fission");
            if (frontier == "primitive") tags.Add("primitive-frontier");
            if (frontier == "plough") tags.Add("plough-frontier");
            double retainedCapital = CultivationEconomy.Footprint(state, camp)
                .Where(id => state.Tiles[id].CultivationOperator == camp.Id)
                .Sum(id => Math.Max(0,
                    LazyEcology.ProjectEcology(state, id, state.Days)
                        .CultivationCapital));
            return new Measurement
            {
                Cultivation = cultivation,
                Husbandry = husbandry,
                FishingQ = fishingQ,
                RetainedFieldCapital = retainedCapital,
                MaintenanceWorkerYears = cultivation ?
                    CultivationEconomy.MaintenanceWorkerRequirement(
                        state, camp) : 0,
                HandToolsPerCapita = Math.Max(0,
                    camp.Inventory.HandTools) / Math.Max(1, camp.Population),
                FarmToolsPerCapita = Math.Max(0,
                    camp.Inventory.FarmTools) / Math.Max(1, camp.Population),
                MaterialsPerCapita = (Math.Max(0, camp.Inventory.Stone) +
                    Math.Max(0, camp.Inventory.Clay) +
                    Math.Max(0, camp.Inventory.Wood)) /
                    Math.Max(1, camp.Population),
                DeliveredTradeTotal = imports + exports,
                DeliveredImportTotal = imports,
                DeliveredExportTotal = exports,
                DeliveredFoodImport = foodImports,
                DeliveredFoodExport = foodExports,
                DeliveredMaterialToolImport = materialToolImports,
                ImportDependence = foodImports / intervalNeed,
                SignedMigration = migration,
                RecentFission = fission != null,
                ParentPopulationRelief = fission?.Daughter ?? 0,
                FieldFrontier = frontier,
                Tags = tags.OrderBy(value => value,
                    StringComparer.Ordinal).ToArray()
            };
        }

        private static bool MaterialOrTool(TradeGood good)
            => good == TradeGood.HandTools || good == TradeGood.FarmTools ||
               good == TradeGood.HousingMaterial || good == TradeGood.Stone ||
               good == TradeGood.Clay || good == TradeGood.Wood;

        private static void ValidateRequestedState(TrajectoryPlan plan,
            ElderObservation observation, string[] actualTags,
            int decisionIndex, bool technologyTransition)
        {
            bool expectedCultivation = SnapshotScenarioFactory
                .HasCultivation(plan.Economy) ||
                plan.Frontier == "technology-transition-cultivation" &&
                technologyTransition;
            bool expectedHusbandry = SnapshotScenarioFactory
                .HasHusbandry(plan.Economy);
            if (observation.Mask.FieldDevelopment != expectedCultivation)
                throw new InvalidOperationException(
                    "Achieved cultivation capability disagrees with requested legal state.");
            if (observation.Capabilities.First(item => item.Key ==
                    "herd-management").Available != expectedHusbandry)
                throw new InvalidOperationException(
                    "Achieved husbandry capability disagrees with request.");
            if (observation.Fishing.Length != plan.FishingGroundTarget)
                throw new InvalidOperationException(
                    "Achieved fishing-ground count disagrees with request.");
            if (plan.FieldQ > 0 && Math.Abs(
                    observation.Fields.CurrentIntensity - plan.FieldQ) > .15)
                throw new InvalidOperationException(
                    "Achieved field q is outside the disclosed requested band.");
            if (plan.HerdSupportRatio > 0 &&
                observation.Herd.Animals <= 0)
                throw new InvalidOperationException(
                    "Requested herd trajectory lost all owned animals.");
            var actual = new HashSet<string>(actualTags,
                StringComparer.Ordinal);
            foreach (string tag in plan.RequestedTags)
                if (!actual.Contains(tag))
                    throw new InvalidOperationException(
                        "Requested stratum was not achieved: " + tag);
            if (plan.Frontier == "technology-transition-cultivation" &&
                decisionIndex > 0 && !technologyTransition)
                throw new InvalidOperationException(
                    "Requested endogenous technology transition did not occur.");
        }

        private static void ValidateSchemaAndMasks(ElderResetResult reset,
            string expectedConfigurationHash, string expectedFingerprint)
        {
            if (reset.ObservationSchemaVersion !=
                    ElderRLContract.ObservationSchemaVersion ||
                reset.ActionSchemaVersion !=
                    ElderRLContract.ActionSchemaVersion ||
                reset.EnvironmentSchemaVersion !=
                    ElderRLContract.EnvironmentSchemaVersion)
                throw new InvalidOperationException(
                    "Reset schema does not match the frozen Elder contract.");
            if (!string.Equals(reset.SimulatorVersion,
                    NativeElderEnvironment.SimulatorVersion,
                    StringComparison.Ordinal) ||
                !string.Equals(reset.BuildHash,
                    NativeElderEnvironment.BuildHash,
                    StringComparison.Ordinal) ||
                !string.Equals(reset.ConfigurationHash,
                    expectedConfigurationHash,
                    StringComparison.Ordinal) ||
                !string.Equals(reset.InitialStateFingerprint,
                    expectedFingerprint, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Reset simulator/build/configuration/fingerprint identity mismatch.");
            var mask = reset.Observation.Mask;
            if (mask.HerdDevelopment && !mask.HerdRetention)
                throw new InvalidOperationException(
                    "Invalid structural mask: herd Development without retained herd.");
            if (!mask.AllowsMovement(-1))
                throw new InvalidOperationException(
                    "Invalid structural movement mask: stay is illegal.");
        }

        private static void ValidateFiniteTensor(ElderTensorObservation tensor)
        {
            if (tensor.Values.Length != ElderRLContract.TensorFeatureCount ||
                tensor.Present.Length != ElderRLContract.TensorFeatureCount)
                throw new InvalidOperationException(
                    "Encoded observation width mismatch.");
            if (tensor.Values.Any(value => float.IsNaN(value) ||
                    float.IsInfinity(value)))
                throw new InvalidOperationException(
                    "Encoded observation contains NaN/Inf.");
        }

        private static ElderResetRequest Request(GenerationPlan plan,
            byte[] snapshot, int camp, int limit,
            string expectedConfigurationHash) =>
            new ElderResetRequest
        {
            Snapshot = snapshot,
            ControlledCampId = camp,
            RandomStream = 0,
            BackgroundPolicy = ElderBackgroundPolicy.FrozenHeuristic,
            Options = new ElderEnvironmentOptions
            {
                EpisodeDecisionLimit = Math.Max(1, limit),
                MovementEnabled = plan.BankKind == "elder-broad-domain" ||
                    plan.BankKind == "elder-broad-domain-diagnostic",
                SyntheticShocksEnabled = false,
                IncludeTrainingCriticSidecar = true
            },
            ExpectedSimulatorVersion = NativeElderEnvironment.SimulatorVersion,
            ExpectedConfigurationHash = expectedConfigurationHash
        };

        private static ElderDirective TrajectoryDirective(
            ElderObservation observation, int step)
            => Directive(observation, step, .15 + .10 * (step % 3));

        private static ElderDirective CalibrationDirective(
            GenerationPlan plan, ElderObservation observation, int step)
        {
            var directive = Directive(observation, step,
                .10 + .15 * (step % 3));
            if ((plan.BankKind != "elder-broad-domain" &&
                    plan.BankKind != "elder-broad-domain-diagnostic") || step != 2)
                return directive;
            for (int candidate = 0; candidate < 7; candidate++)
                if (observation.Mask.AllowsMovement(candidate))
                {
                    directive.MovementCandidate = candidate;
                    break;
                }
            return directive;
        }

        private static ElderDirective Directive(ElderObservation observation,
            int step, double development)
        {
            var mask = observation.Mask;
            return new ElderDirective
            {
                RetainedFieldOperation = mask.FieldRetention ? 1 : 0,
                RetainedHerdFraction = mask.HerdRetention ? 1 : 0,
                DevelopmentBudget = mask.AllowsDevelopment() ?
                    development : 0,
                FieldDevelopmentShare = mask.AllowsDevelopmentSplit() ?
                    new[] { .2, .5, .8 }[step % 3] :
                    mask.FieldDevelopment ? 1 :
                    mask.HerdDevelopment ? 0 : .5,
                MovementCandidate = -1
            };
        }

        private static Dictionary<string, object> SelectionMetadata(
            TrajectoryPlan plan, string siteEvidence,
            Dictionary<string, object> additional = null)
        {
            var metadata = new Dictionary<string, object>
            {
                ["trajectory_id"] = plan.TrajectoryId,
                ["requested_field_q"] = plan.FieldQ,
                ["requested_herd_support_ratio"] =
                    plan.HerdSupportRatio,
                ["requested_reserve_intervals"] =
                    plan.ReserveIntervals,
                ["requested_fishing_ground_count"] =
                    plan.FishingGroundTarget,
                ["requested_site_support_quantile"] =
                    plan.SiteSupportQuantile,
                ["requested_tags"] = plan.RequestedTags,
                ["frontier_request"] = plan.Frontier,
                ["causal_variant"] = plan.CausalVariant,
                ["source_policy_identity"] = plan.SourceController,
                ["site_selection_evidence"] = siteEvidence,
                ["historical_or_consumed_diagnostic"] =
                    plan.HistoricalOrConsumedDiagnostic
            };
            if (additional != null)
                foreach (var item in additional)
                    metadata[item.Key] = item.Value;
            return metadata;
        }

        private static RejectedCandidate Reject(TrajectoryPlan plan,
            Exception error,
            Dictionary<string, object> selectionMetadata = null)
            => new RejectedCandidate
        {
            CandidateId = plan.TrajectoryId,
            ScenarioFamily = plan.ScenarioFamily,
            ScenarioVersion = plan.ScenarioVersion,
            WorldSeed = plan.WorldSeed,
            TrajectoryFamily = plan.TrajectoryFamily,
            LineageId = plan.LineageId,
            CounterfactualGroupId = plan.CounterfactualGroupId,
            RequestedPartition = plan.Partition,
            ReasonCode = error.Message.Contains("No physically compatible site") ||
                    error.Message.Contains("Requested domestic herd support ratio")
                ? "requested-stratum-unreachable"
                : error.Message.Contains("Requested stratum was not achieved")
                    ? "requested-stratum-not-achieved"
                    : "native-generation-rejected",
            Reason = error.GetType().Name + ": " + error.Message,
            SelectionMetadata = SelectionMetadata(plan, "not-selected",
                selectionMetadata)
        };

        private static void RecordConfigurationHash(GeneratorResult output,
            string configurationHash)
        {
            if (string.IsNullOrEmpty(output.ConfigurationHash))
            {
                output.ConfigurationHash = configurationHash;
                return;
            }
            if (!string.Equals(output.ConfigurationHash, configurationHash,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Accepted candidates did not share one native " +
                    "configuration hash.");
        }

        private static string SnapshotId(TrajectoryPlan plan, int step,
            string fingerprint)
        {
            string raw = (plan.ScenarioFamily + "-" + plan.WorldSeed + "-" +
                plan.TrajectoryId + "-d" + step + "-" +
                fingerprint.Substring(0, 12)).ToLowerInvariant();
            return new string(raw.Select(character =>
                char.IsLetterOrDigit(character) || character == '-' ?
                    character : '-').ToArray());
        }

        private static double MemoryAgeYears(WorldState state, Camp camp)
            => camp.ElderMemory?.Initialized == true ? Math.Max(0,
                state.Days - camp.ElderMemory.InitializedDays) /
                state.Config.DaysPerYear : 0;

        private static double? Ratio(double numerator, double denominator)
            => denominator > 0 ? numerator / denominator : (double?)null;

        private static string ObservationSchemaHash()
        {
            byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(new
            {
                registry = ElderObservationSchema.Registry,
                capability_rows = ElderObservationSchema.CapabilityRows,
                commodity_rows = ElderObservationSchema.CommodityRows,
                memory_rows = ElderObservationSchema.MemoryRows,
                memory_time_constants_years =
                    ElderInstitutionMemory.TimeConstantsYears
            }, Json);
            return Sha256(bytes);
        }

        private static string Sha256(byte[] bytes)
        {
            using var hash = SHA256.Create();
            return string.Concat(hash.ComputeHash(bytes).Select(value =>
                value.ToString("x2", CultureInfo.InvariantCulture)));
        }

        private static void ValidatePlan(GenerationPlan plan)
        {
            if (plan.ConfigSchemaVersion != "snapshot-bank-generation-v1")
                throw new InvalidDataException(
                    "Unsupported snapshot-bank generation config schema.");
            if (plan.NativeSimulatorSha != FrozenSimulatorSha)
                throw new InvalidDataException(
                    "Generation plan does not pin the frozen simulator SHA.");
            if (plan.Trajectories.Length == 0 ||
                plan.CalibrationRolloutSteps < 1 ||
                plan.CalibrationRolloutSteps > 5)
                throw new InvalidDataException(
                    "Generation plan is empty or has an invalid rollout length.");
            int snapshots = plan.Trajectories.Sum(item =>
                item.SnapshotSteps.Length);
            if (plan.BankKind == "stage0" &&
                (snapshots < 24 || snapshots > 48))
                throw new InvalidDataException(
                    "Stage-0 plan must request 24-48 snapshots.");
            if (plan.BankKind == "cultivation-pilot" &&
                (snapshots < 200 || snapshots > 800))
                throw new InvalidDataException(
                    "Cultivation pilot plan must request 200-800 candidates.");
            if (plan.BankKind == "elder-broad-domain" &&
                (snapshots < 400 || snapshots > 800))
                throw new InvalidDataException(
                    "Broad-domain plan must request 400-800 candidates.");
            if (plan.BankKind == "elder-broad-domain-diagnostic" &&
                (snapshots < 2 || snapshots > 48))
                throw new InvalidDataException(
                    "Broad-domain diagnostic plan must request 2-48 candidates.");
            if (plan.BankKind != "stage0" &&
                plan.BankKind != "cultivation-pilot" &&
                plan.BankKind != "elder-broad-domain" &&
                plan.BankKind != "elder-broad-domain-diagnostic")
                throw new InvalidDataException(
                    "Unsupported snapshot-bank kind.");
            if (plan.F15SharedCaptureHorizonSteps < 0 ||
                plan.F15SharedCaptureHorizonSteps > 5)
                throw new InvalidDataException(
                    "F15 shared capture horizon must be between zero and five.");
            if (plan.F15SharedCaptureHorizonSteps > 0)
            {
                if (plan.BankKind != "elder-broad-domain" &&
                    plan.BankKind != "elder-broad-domain-diagnostic")
                    throw new InvalidDataException(
                        "Event-aligned F15 capture is limited to broad-domain " +
                        "banks and diagnostics.");
                foreach (var group in plan.Trajectories
                    .Where(item => item.ScenarioFamily == "F15" &&
                        (item.CausalVariant == "mild-import" ||
                            item.CausalVariant == "surplus-export"))
                    .GroupBy(item => item.CounterfactualGroupId,
                        StringComparer.Ordinal))
                    ValidateEventAlignedPair(group.ToArray());
            }
        }
    }
}
