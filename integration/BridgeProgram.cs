using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using Anthropocene.Simulation;

namespace Anthropocene.RLBridge
{
    internal static class BridgeProgram
    {
        private static readonly JsonSerializerOptions Json =
            new JsonSerializerOptions
            {
                IncludeFields = true,
                PropertyNameCaseInsensitive = true,
                WriteIndented = false
            };

        private static NativeElderEnvironment environment;
        private static int controlledCampId = -1;

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
                    if (op == "close")
                    {
                        Write(new { ok = true, op });
                        return 0;
                    }
                    if (op == "schema")
                    {
                        Write(new
                        {
                            ok = true,
                            op,
                            observationSchemaVersion =
                                ElderRLContract.ObservationSchemaVersion,
                            actionSchemaVersion =
                                ElderRLContract.ActionSchemaVersion,
                            environmentSchemaVersion =
                                ElderRLContract.EnvironmentSchemaVersion,
                            criticSidecarSchemaVersion =
                                ElderRLContract.CriticSidecarSchemaVersion,
                            tensorFeatureCount =
                                ElderRLContract.TensorFeatureCount,
                            simulatorVersion =
                                NativeElderEnvironment.SimulatorVersion
                        });
                        continue;
                    }
                    if (op == "reset")
                    {
                        var request = root.GetProperty("request")
                            .Deserialize<ElderResetRequest>(Json);
                        if (request == null)
                            throw new ArgumentException("reset request is null");
                        environment = new NativeElderEnvironment();
                        var reset = environment.Reset(request);
                        controlledCampId = reset.ControlledCampId;
                        Write(new
                        {
                            ok = true,
                            op,
                            result = new
                            {
                                reset.ObservationSchemaVersion,
                                reset.ActionSchemaVersion,
                                reset.EnvironmentSchemaVersion,
                                reset.SimulatorVersion,
                                reset.BuildHash,
                                reset.ConfigurationHash,
                                reset.InitialStateFingerprint,
                                reset.ControlledCampId,
                                reset.ControlledPlayer,
                                reset.FixtureIdentity,
                                reset.Seed,
                                reset.RandomStream,
                                reset.TrainingCriticSidecar,
                                StateDiagnostics = CaptureStateDiagnostics(
                                    environment.CaptureSnapshot(),
                                    reset.ControlledCampId,
                                    reset.Observation)
                            },
                            tensor = ElderObservationSchema.Encode(
                                reset.Observation)
                        });
                        continue;
                    }
                    if (op == "step")
                    {
                        if (environment == null)
                            throw new InvalidOperationException(
                                "reset must precede step");
                        var directive = root.GetProperty("directive")
                            .Deserialize<ElderDirective>(Json);
                        if (directive == null)
                            throw new ArgumentException("directive is null");
                        long started = Stopwatch.GetTimestamp();
                        var step = environment.Step(directive);
                        double nativeStepSeconds = Stopwatch.GetElapsedTime(
                            started).TotalSeconds;
                        WriteStep(op, step, nativeStepSeconds,
                            "learned-policy");
                        continue;
                    }
                    if (op == "option_a_step")
                    {
                        if (environment == null)
                            throw new InvalidOperationException(
                                "reset must precede option_a_step");
                        var committed = SaveCodec.Load(
                            environment.CaptureSnapshot());
                        var camp = committed.Camps.FirstOrDefault(item =>
                            item.Id == controlledCampId) ?? throw new
                            InvalidOperationException(
                                "controlled Option-A camp is no longer active");
                        var observation = new ElderObservationBuilder().Build(
                            committed, controlledCampId, false, out _);
                        var directive = OptionAElderPolicy.FromCommittedState(
                            committed, camp.Id).Decide(observation);
                        long started = Stopwatch.GetTimestamp();
                        var step = environment.Step(directive);
                        double nativeStepSeconds = Stopwatch.GetElapsedTime(
                            started).TotalSeconds;
                        WriteStep(op, step, nativeStepSeconds,
                            "option-a-through-shared-compiler");
                        continue;
                    }
                    if (op == "verify_replay")
                    {
                        if (environment == null)
                            throw new InvalidOperationException(
                                "reset must precede replay verification");
                        NativeElderEnvironment.VerifyReplay(
                            environment.ReplayLog);
                        Write(new
                        {
                            ok = true,
                            op,
                            steps = environment.ReplayLog.Steps.Count,
                            finalStateFingerprint = environment.ReplayLog.Steps
                                .Count == 0 ?
                                environment.ReplayLog.InitialStateFingerprint :
                                environment.ReplayLog.Steps[
                                    environment.ReplayLog.Steps.Count - 1]
                                    .PostStepFingerprint
                        });
                        continue;
                    }
                    throw new ArgumentException("Unknown bridge op: " + op);
                }
                catch (Exception error)
                {
                    Write(new
                    {
                        ok = false,
                        errorType = error.GetType().Name,
                        message = error.Message
                    });
                }
            }
            return 0;
        }

        private static void Write(object value)
        {
            Console.WriteLine(JsonSerializer.Serialize(value, Json));
            Console.Out.Flush();
        }

        private static void WriteStep(string op, ElderStepResult step,
            double nativeStepSeconds, string policy)
        {
            Write(new
            {
                ok = true,
                op,
                result = new
                {
                    step.ScalarReward,
                    step.ElapsedYears,
                    step.Terminated,
                    step.Truncated,
                    step.StateFingerprint,
                    step.Reward,
                    Requested = step.Decision?.Requested,
                    Accepted = step.Decision?.Accepted,
                    Realized = step.Decision?.Realized,
                    Interventions = step.Decision?.Interventions,
                    step.TrainingCriticSidecar,
                    step.Events,
                    NativeStepSeconds = nativeStepSeconds,
                    Policy = policy,
                    StateDiagnostics = CaptureStateDiagnostics(
                        environment.CaptureSnapshot(), controlledCampId,
                        step.Observation)
                },
                tensor = step.Observation == null ? null :
                    ElderObservationSchema.Encode(step.Observation)
            });
        }

        private static object CaptureStateDiagnostics(byte[] snapshot,
            int campId, ElderObservation observation)
        {
            var state = SaveCodec.Load(snapshot);
            var camp = state.Camps.FirstOrDefault(item => item.Id == campId);
            if (camp == null || observation == null)
                return new
                {
                    Active = false,
                    Population = 0,
                    PhysicalYears = state.Days / state.Config.DaysPerYear
                };

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
            var demographic = recent?.Demographics?.FirstOrDefault(item =>
                item.Camp == camp.Id);
            var fission = (recent?.Fissions ?? new List<FissionReport>())
                .LastOrDefault(item => item.ParentCamp == camp.Id);
            double intervalNeed = Math.Max(1e-12,
                Math.Max(1, demographic?.StartingPopulation ??
                    camp.Population) * state.Config.Consumption *
                Math.Max(0, recent?.Duration ?? 0) / state.Config.BaseDays);
            double retainedCapital = CultivationEconomy.Footprint(state, camp)
                .Where(id => state.Tiles[id].CultivationOperator == camp.Id)
                .Sum(id => Math.Max(0,
                    LazyEcology.ProjectEcology(state, id, state.Days)
                        .CultivationCapital));
            return new
            {
                Active = true,
                PhysicalYears = state.Days / state.Config.DaysPerYear,
                Population = observation.Core.Population,
                observation.Core.QoL,
                FoodNeedsMet = observation.Core.FoodNeedsMet,
                WaterNeedsMet = observation.Core.WaterNeedsMet,
                ClothingCoverage = observation.Core.ClothingCoverage,
                HousingCoverage = observation.Core.HousingCoverage,
                FoodReserveYears = observation.Core.FoodReserveYears,
                FieldQ = observation.Fields.CurrentIntensity,
                FieldOperation = observation.Fields.CurrentOperation,
                RetainedFieldCapital = retainedCapital,
                MaintenanceWorkerYears =
                    observation.Fields.MaintenanceWorkerYears,
                HandToolsPerCapita = Math.Max(0, camp.Inventory.HandTools) /
                    Math.Max(1, camp.Population),
                FarmToolsPerCapita = Math.Max(0, camp.Inventory.FarmTools) /
                    Math.Max(1, camp.Population),
                MaterialsPerCapita = (Math.Max(0, camp.Inventory.Stone) +
                    Math.Max(0, camp.Inventory.Clay) +
                    Math.Max(0, camp.Inventory.Wood)) /
                    Math.Max(1, camp.Population),
                DeliveredTrade = imports + exports,
                DeliveredImports = imports,
                DeliveredExports = exports,
                DeliveredFoodImports = foodImports,
                DeliveredFoodExports = foodExports,
                DeliveredMaterialToolImports = materialToolImports,
                ImportDependence = foodImports / intervalNeed,
                SignedMigration = migration,
                FissionPressure = observation.Fission.Pressure,
                FissionExitOpportunity =
                    observation.Fission.ExitOpportunityPressure,
                FissionEvent = fission != null,
                FissionDaughterPopulation = fission?.Daughter ?? 0,
                FissionDaughterQoL = fission?.DaughterQoL ?? 0,
                FissionParentPopulationBefore = fission?.ParentBefore ?? 0,
                FissionParentPopulationRelief = fission?.Daughter ?? 0,
                DemographicStartingPopulation =
                    demographic?.StartingPopulation ?? camp.Population,
                DemographicEndingPopulation =
                    demographic?.EndingPopulation ?? camp.Population,
                DemographicDeaths = demographic?.Deaths ?? 0,
                DemographicImmigrated = demographic?.Immigrated ?? 0,
                DemographicEmigrated = demographic?.Emigrated ?? 0,
                DemographicSplitOff = demographic?.SplitOff ?? 0
            };
        }

        private static bool MaterialOrTool(TradeGood good)
            => good == TradeGood.HandTools || good == TradeGood.FarmTools ||
               good == TradeGood.HousingMaterial || good == TradeGood.Stone ||
               good == TradeGood.Clay || good == TradeGood.Wood;
    }
}
