using System;
using System.Collections.Generic;
using System.Linq;

namespace Anthropocene.Simulation
{
    public sealed class AutonomousElderReplayStep
    {
        public Dictionary<int, AutonomousElderAction> Actions;
        public Dictionary<int, List<Order>> Background;
        public Dictionary<int, int[]> Staffing;
        public string Fingerprint;
    }
    public sealed class AutonomousElderReplay
    {
        public int SchemaVersion = AutonomousElderContract.ReplayVersion;
        public int ActionVersion = AutonomousElderContract.ActionVersion;
        public int ObservationVersion = AutonomousElderContract.ObservationVersion;
        public int EnvironmentVersion = AutonomousElderContract.EnvironmentVersion;
        public string BuildHash, ConfigurationHash, InitialFingerprint;
        public byte[] Snapshot;
        public int[] Controlled;
        public int DecisionLimit;
        public List<AutonomousElderReplayStep> Steps = new List<AutonomousElderReplayStep>();
    }
    public sealed class AutonomousElderTelemetry
    {
        public Dictionary<int, int> Population, Moved, SettledTrades, DaughterSettlements, Immigrated, Emigrated;
        public Dictionary<int, double> FoodMet, Quality;
        public int Trades, Fissions, Migrants;
    }
    public sealed class AutonomousElderStep
    {
        public Dictionary<int, AutonomousElderObservation> Observations;
        public Dictionary<int, double> Rewards;
        public Dictionary<int, int[]> Staffing;
        public AutonomousElderTelemetry Telemetry;
        public double ElapsedYears;
        public bool Terminated, Truncated;
        public string Fingerprint;
    }

    public sealed class AutonomousElderEnvironment
    {
        private WorldState world;
        private int[] controlled;
        private int limit, steps;
        private bool ended;
        private readonly PhaseExecution execution;
        private readonly Action<WorldState, int[]> initializeHistory;
        private readonly int actionVersion = 3;
        internal WorldState State => world;
        private Dictionary<int, AutonomousElderObservation> observations;
        public AutonomousElderReplay Replay { get; private set; }
        public AutonomousElderEnvironment(PhaseExecution execution = null) { this.execution = execution ?? new PhaseExecution(); }
        internal AutonomousElderEnvironment(PhaseExecution execution, Action<WorldState, int[]> initializeHistory, int actionVersion = 3)
        { this.execution = execution ?? new PhaseExecution(); this.initializeHistory = initializeHistory;
            if (actionVersion != 3 && actionVersion != 4) throw new ArgumentException("Unsupported authority version.");
            this.actionVersion = actionVersion; }
        public byte[] CaptureSnapshot() => world == null ? throw new InvalidOperationException("Reset required.") : SaveCodec.Save(world);
        public Dictionary<int, AutonomousElderObservation> Reset(byte[] snapshot, int[] controlledCamps, int decisionLimit = 32)
        {
            if (snapshot == null || controlledCamps == null || controlledCamps.Length == 0 || controlledCamps.Distinct().Count() != controlledCamps.Length || decisionLimit < 1)
                throw new ArgumentException("Snapshot, distinct controlled camps and positive limit required.");
            var candidate = SaveCodec.Load((byte[])snapshot.Clone());
            foreach (int id in controlledCamps)
            {
                var c = candidate.Camps.FirstOrDefault(x => x.Id == id);
                if (c == null || c.Population <= 0 || !CultivationEconomy.EconomyEnabled(candidate, c)) throw new ArgumentException("Controlled camp lacks active economy.");
            }
            if (candidate.AutonomousReceipts != null && initializeHistory == null)
                throw new ArgumentException("An autonomous history snapshot requires environment5.");
            if (actionVersion == 3 && candidate.AutonomousReceipts?.Values.Any(r => r.SchemaVersion == 2) == true)
                throw new ArgumentException("Action4 history cannot be downgraded to action3.");
            initializeHistory?.Invoke(candidate, controlledCamps);
            world = candidate; controlled = controlledCamps.OrderBy(x => x).ToArray(); limit = decisionLimit; steps = 0; ended = false;
            observations = Observe();
            Replay = new AutonomousElderReplay { Snapshot = SaveCodec.Save(world), Controlled = (int[])controlled.Clone(), DecisionLimit = limit,
                BuildHash = NativeElderEnvironment.BuildHash, ConfigurationHash = SaveCodec.ConfigurationHash(world.Config), InitialFingerprint = SaveCodec.Fingerprint(world) };
            return CopyObservations();
        }
        private Dictionary<int, AutonomousElderObservation> Observe()
            => world.Camps.Where(c => controlled.Contains(c.Id) && c.Population > 0).OrderBy(c => c.Id)
                .ToDictionary(c => c.Id, c => { var o = AutonomousElderCompiler.Observe(world, c);
                    if (actionVersion == 4) o.SchemaVersion = 6; return o; });

        // Server address metadata: only the observer's committed legal known
        // contacts. Never query the target's current state or submitted action.
        internal int KnownCounterpartyOrdinal(int observer, int target)
        {
            if (observations == null || !observations.TryGetValue(observer, out var own))
                throw new ArgumentException("A living controlled observer is required.");
            int ordinal = Array.IndexOf(own.CounterpartyIds, target);
            return ordinal >= 0 && own.CounterpartyMask[ordinal] ? ordinal : -1;
        }
        private Dictionary<int, AutonomousElderObservation> CopyObservations()
            => observations.ToDictionary(x => x.Key, x => new AutonomousElderObservation
            {
                SchemaVersion = x.Value.SchemaVersion,
                MovementMask = (bool[])x.Value.MovementMask.Clone(),
                Context = (double[])x.Value.Context.Clone(), WorkforceMask = (bool[])x.Value.WorkforceMask.Clone(),
                ActivityMask = x.Value.ActivityMask.Select(row => (bool[])row.Clone()).ToArray(),
                ControlMask = (bool[])x.Value.ControlMask.Clone(), CounterpartyMask = (bool[])x.Value.CounterpartyMask.Clone(),
                GoodMask = (bool[])x.Value.GoodMask.Clone(), Stocks = (double[])x.Value.Stocks.Clone()
            });
        public AutonomousElderStep Step(IReadOnlyDictionary<int, AutonomousElderAction> submitted)
            => StepCore(submitted, null);
        internal AutonomousElderStep StepWithBackground(IReadOnlyDictionary<int, AutonomousElderAction> submitted,
            Dictionary<int, List<Order>> background) => StepCore(submitted, background);
        private AutonomousElderStep StepCore(IReadOnlyDictionary<int, AutonomousElderAction> submitted, Dictionary<int, List<Order>> replayBackground)
        {
            if (world == null || ended) throw new InvalidOperationException("Reset required.");
            if (submitted == null || !submitted.Keys.OrderBy(x => x).SequenceEqual(observations.Keys.OrderBy(x => x))) throw new ArgumentException("Exactly one action per living controlled camp at this WEGO boundary is required.");
            foreach (var pair in submitted) AutonomousElderCompiler.Validate(pair.Value, observations[pair.Key]);
            var actions = submitted.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value.Copy());
            var background = replayBackground ?? AiPlanning.Decide(world, execution).Orders.ToDictionary(x => x.Key, x => x.Value.Select(o => o.Copy()).ToList());
            // Keep hunter/other-camp decisions, replacing only controlled camp orders.
            var orders = background.ToDictionary(x => x.Key, x => x.Value.Where(o => !controlled.Contains(o.Entity)).Select(o => o.Copy()).ToList());
            var prior = world.Camps.Where(c => actions.ContainsKey(c.Id)).ToDictionary(c => c.Id, c => c.Population);
            var priorTiles = world.Camps.Where(c => actions.ContainsKey(c.Id)).ToDictionary(c => c.Id, c => c.Tile);
            var staffing = actions.ToDictionary(x => x.Key, x => AutonomousElderCompiler.Apportion(x.Value.Workforce, prior[x.Key]));
            foreach (var c in world.Camps.Where(c => actions.ContainsKey(c.Id)))
            {
                if (!orders.ContainsKey(c.Player)) orders[c.Player] = new List<Order>();
                int movement = actions[c.Id].Movement;
                orders[c.Player].Add(movement == 0 ? new Order(OrderKind.NoOp, c.Id) :
                    new Order(OrderKind.Move, c.Id, destination: observations[c.Id].SiteTiles[movement - 1]));
            }
            var savedBackground = orders.ToDictionary(x => x.Key, x => x.Value.Select(o => o.Copy()).ToList());
            var prepared = TurnPlanner.PrepareAutonomous(world, orders, actions, observations, execution);
            var record = new TurnResolver(execution).Resolve(world, prepared);
            steps++;
            observations = Observe();
            bool terminated = observations.Count == 0, truncated = !terminated && steps >= limit;
            ended = terminated || truncated;
            string fingerprint = SaveCodec.Fingerprint(world);
            Replay.Steps.Add(new AutonomousElderReplayStep { Actions = actions, Background = savedBackground, Staffing = staffing.ToDictionary(x => x.Key, x => (int[])x.Value.Clone()), Fingerprint = fingerprint });
            double years = record.Duration / world.Config.DaysPerYear;
            var camps = world.Camps.ToDictionary(c => c.Id);
            return new AutonomousElderStep { Observations = CopyObservations(), Staffing = staffing, ElapsedYears = years,
                Telemetry = new AutonomousElderTelemetry {
                    Population = prior.ToDictionary(x => x.Key, x => camps.TryGetValue(x.Key, out var c) ? c.Population : 0),
                    Moved = priorTiles.ToDictionary(x => x.Key, x => camps.TryGetValue(x.Key, out var c) && c.Tile != x.Value ? 1 : 0),
                    FoodMet = prior.ToDictionary(x => x.Key, x => camps.TryGetValue(x.Key, out var c) ? c.FoodNeedsMetRatio : 0),
                    Quality = prior.ToDictionary(x => x.Key, x => camps.TryGetValue(x.Key, out var c) ? c.QoL : 0),
                    SettledTrades = prior.ToDictionary(x => x.Key, x => record.Trades.Count(t => t.Requester == x.Key || t.Responder == x.Key)),
                    DaughterSettlements = prior.ToDictionary(x => x.Key, x => record.Fissions.Count(f => f.ParentCamp == x.Key)),
                    Immigrated = prior.ToDictionary(x => x.Key, x => record.Migrations.Where(m => m.DestinationCamp == x.Key).Sum(m => m.Migrants)),
                    Emigrated = prior.ToDictionary(x => x.Key, x => record.Migrations.Where(m => m.SourceCamp == x.Key).Sum(m => m.Migrants)),
                    Trades = record.Trades.Count, Fissions = record.Fissions.Count, Migrants = record.Migrations.Sum(x => x.Migrants)
                },
                Rewards = prior.ToDictionary(x => x.Key, x => ElderRewardContract.Physical(x.Value, camps.TryGetValue(x.Key, out var c) ? ElderRLContract.Unit(c.QoL) : 0, years).UndiscountedIntervalWelfare),
                Terminated = terminated, Truncated = truncated, Fingerprint = fingerprint };
        }
        public static void VerifyReplay(AutonomousElderReplay replay, PhaseExecution execution = null)
        {
            if (replay == null || replay.SchemaVersion != AutonomousElderContract.ReplayVersion || replay.ActionVersion != AutonomousElderContract.ActionVersion ||
                replay.ObservationVersion != AutonomousElderContract.ObservationVersion || replay.EnvironmentVersion != AutonomousElderContract.EnvironmentVersion || replay.BuildHash != NativeElderEnvironment.BuildHash)
                throw new ArgumentException("Replay version/build mismatch.");
            var env = new AutonomousElderEnvironment(execution);
            env.Reset(replay.Snapshot, replay.Controlled, replay.DecisionLimit);
            if (env.Replay.ConfigurationHash != replay.ConfigurationHash || env.Replay.InitialFingerprint != replay.InitialFingerprint) throw new ArgumentException("Replay initial boundary mismatch.");
            foreach (var expected in replay.Steps)
            {
                var result = env.StepCore(expected.Actions, expected.Background);
                if (result.Fingerprint != expected.Fingerprint || !result.Staffing.Keys.OrderBy(x => x).SequenceEqual(expected.Staffing.Keys.OrderBy(x => x)) ||
                    result.Staffing.Any(x => !x.Value.SequenceEqual(expected.Staffing[x.Key]))) throw new InvalidOperationException("Autonomous replay divergence.");
            }
        }
    }
}
