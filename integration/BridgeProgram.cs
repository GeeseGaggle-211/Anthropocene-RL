using System;
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
                                reset.TrainingCriticSidecar
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
                        var step = environment.Step(directive);
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
                                step.Events
                            },
                            tensor = ElderObservationSchema.Encode(
                                step.Observation)
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
    }
}
