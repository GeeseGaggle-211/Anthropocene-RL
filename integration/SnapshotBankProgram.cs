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
            foreach (var trajectory in plan.Trajectories.OrderBy(item =>
                         item.TrajectoryId, StringComparer.Ordinal))
            {
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
            var reset = environment.Reset(Request(fixture,
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
                    if (string.IsNullOrWhiteSpace(output.ConfigurationHash))
                        output.ConfigurationHash = configHash;
                    else if (!string.Equals(output.ConfigurationHash,
                            configHash, StringComparison.Ordinal))
                        throw new InvalidOperationException(
                            "Native configuration hash drifted within one bank.");
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

        private static CandidateEnvelope AcceptCandidate(GenerationPlan plan,
            TrajectoryPlan trajectory, int decisionIndex, byte[] snapshot,
            int controlledCamp, string siteEvidence,
            bool technologyTransition, string snapshotDirectory,
            string codeHash, string schemaHash,
            out string configurationHash)
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
            var reset = environment.Reset(Request(canonical,
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
                    current, step));
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
            var selection = SelectionMetadata(trajectory, siteEvidence);
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
            public string FieldFrontier;
            public string[] Tags;
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
            return new Measurement
            {
                Cultivation = cultivation,
                Husbandry = husbandry,
                FishingQ = fishingQ,
                FieldFrontier = frontier,
                Tags = tags.OrderBy(value => value,
                    StringComparer.Ordinal).ToArray()
            };
        }

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

        private static ElderResetRequest Request(byte[] snapshot, int camp,
            int limit, string expectedConfigurationHash) =>
            new ElderResetRequest
        {
            Snapshot = snapshot,
            ControlledCampId = camp,
            RandomStream = 0,
            BackgroundPolicy = ElderBackgroundPolicy.FrozenHeuristic,
            Options = new ElderEnvironmentOptions
            {
                EpisodeDecisionLimit = Math.Max(1, limit),
                MovementEnabled = false,
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
            ElderObservation observation, int step)
            => Directive(observation, step, .10 + .15 * (step % 3));

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
            TrajectoryPlan plan, string siteEvidence)
            => new Dictionary<string, object>
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
                ["source_policy_identity"] = plan.SourceController,
                ["site_selection_evidence"] = siteEvidence,
                ["historical_or_consumed_diagnostic"] =
                    plan.HistoricalOrConsumedDiagnostic
            };

        private static RejectedCandidate Reject(TrajectoryPlan plan,
            Exception error) => new RejectedCandidate
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
            SelectionMetadata = SelectionMetadata(plan, "not-selected")
        };

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
            if (snapshots < 24 || snapshots > 48)
                throw new InvalidDataException(
                    "Stage-0 plan must request 24-48 snapshots.");
        }
    }
}
