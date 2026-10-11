using System;
using System.Collections.Generic;
using Anthropocene.Simulation;

namespace Anthropocene.SnapshotBank
{
    internal sealed class WorldGenerationConfig
    {
        public int Width, Height, Players, StartSeparation;
        public double AnnualHumanGrowthRate;
    }

    internal sealed class PartitionConfig
    {
        public string Salt;
        public double TrainFraction, DevelopmentFraction, HeldoutFraction;
    }

    internal sealed class TrajectoryPlan
    {
        public string TrajectoryId, ScenarioFamily, ScenarioVersion;
        public int WorldSeed;
        public string TrajectoryFamily, LineageId, CounterfactualGroupId;
        public string SourceController, Classification, Partition;
        public bool HistoricalOrConsumedDiagnostic;
        public int[] SnapshotSteps = Array.Empty<int>();
        public string Economy;
        public int Population;
        public double FieldQ, HerdSupportRatio, ReserveIntervals;
        public int FishingGroundTarget;
        public double SiteSupportQuantile, ToolAbundance, AmenityCoverage;
        public string Frontier, CausalVariant;
        public string[] RequestedTags = Array.Empty<string>();
    }

    internal sealed class GenerationPlan
    {
        public string ConfigSchemaVersion, BankKind, BankId, BankVersion,
            NativeGeneratorVersion, NativeSimulatorSha,
            GeneratorConfigSha256;
        public int GenerationSeed, CalibrationRolloutSteps;
        public bool VerifyDeterministicRegeneration;
        public double LocalMovementRadius;
        public Dictionary<string, double> CapabilityExperienceThresholds =
            new Dictionary<string, double>();
        public WorldGenerationConfig World = new WorldGenerationConfig();
        public PartitionConfig Partition = new PartitionConfig();
        public TrajectoryPlan[] Trajectories = Array.Empty<TrajectoryPlan>();
    }

    internal sealed class NativeObservationSample
    {
        public string Source;
        public ElderTensorObservation Tensor;
    }

    internal sealed class CandidateEnvelope
    {
        public object Record;
        public string SnapshotFile;
        public NativeObservationSample[] CalibrationObservations =
            Array.Empty<NativeObservationSample>();
    }

    internal sealed class RejectedCandidate
    {
        public string CandidateId, ScenarioFamily, ScenarioVersion;
        public int WorldSeed;
        public string TrajectoryFamily, LineageId, CounterfactualGroupId,
            RequestedPartition, ReasonCode, Reason;
        public Dictionary<string, object> SelectionMetadata =
            new Dictionary<string, object>();
    }

    internal sealed class GeneratorResult
    {
        public string Status, NativeSimulatorSha, NativeSimulatorVersion,
            NativeBuildHash, ConfigurationHash, ObservationSchemaHash,
            GeneratorCodeSha256, GeneratorVersion,
            GeneratorConfigSha256;
        public int ObservationSchemaVersion, ActionSchemaVersion,
            EnvironmentSchemaVersion, CriticSidecarSchemaVersion,
            TensorFeatureCount;
        public List<CandidateEnvelope> Candidates =
            new List<CandidateEnvelope>();
        public List<RejectedCandidate> RejectedCandidates =
            new List<RejectedCandidate>();
    }

    internal sealed class ScenarioBuildResult
    {
        public WorldState World;
        public int ControlledCampId;
        public string SiteSelectionEvidence;
    }
}
