using System;
using System.Collections.Generic;
using System.Linq;
using Anthropocene.Simulation;

namespace Anthropocene.RLBridge
{
    // Explicit analysis sidecar. No actor or value encoder consumes it.
    internal static class AutonomousOwnerDiagnostics
    {
        internal static object Capture(byte[] snapshot, int[] controlled)
        {
            var world = SaveCodec.Load(snapshot); var last = world.History.LastOrDefault();
            return controlled.ToDictionary(id => id, id => {
                var camp = world.Camps.FirstOrDefault(c => c.Id == id);
                var herds = last?.ManagedHerds.Where(h => h.Camp == id).ToArray() ?? Array.Empty<ManagedHerdReport>();
                var receipt = world.AutonomousReceipts[id];
                var observed = camp == null || camp.Population <= 0 ? null : AutonomousElderHistoryEnvironment.Observe(world, id);
                return new {
                    schemaVersion = 1, population = camp?.Population ?? 0, qol = camp?.QoL ?? 0,
                    needsProfile = world.Config.Neolithic.SubsistenceNeedsCovered ? "subsistence-needs-v13" : "legacy",
                    automaticNeedsResidents = camp == null ? 0 : SubsistenceNeeds.CoveredResidents(world, camp, camp.Population),
                    physicalClothingPerCapita = camp == null ? 0 : SubsistenceNeeds.PhysicalClothingFraction(camp),
                    clothingCoverage = camp?.ClothPerCapita ?? 0, housingCoverage = camp?.HousingPerCapita ?? 0,
                    knownControlledCounterpartyOrdinals = observed == null ? Array.Empty<int>() :
                        Enumerable.Range(0, observed.CounterpartyIds.Length).Where(n =>
                            observed.CounterpartyMask[n] && controlled.Contains(observed.CounterpartyIds[n])).ToArray(),
                    stocks = camp == null ? new double[11] : Trade.Registry.Select(g => Trade.Held(camp, g.Id)).ToArray(),
                    startingAnimals = herds.Sum(h => h.StartingAnimals), survivingAnimals = herds.Sum(h => h.SurvivingAnimals),
                    newbornAnimals = herds.Sum(h => h.NewbornAnimals), slaughtered = herds.Sum(h => h.Slaughtered),
                    acceptedAnimalFood = herds.Sum(h => h.AcceptedFood), slaughterFood = herds.Sum(h => h.SlaughterFood),
                    aquaticHarvest = last?.Transfers.Where(t => t.Entity == id &&
                        t.Kind.IndexOf("Fish", StringComparison.OrdinalIgnoreCase) >= 0).Sum(t => t.Food) ?? 0,
                    animalStarvationDeaths = herds.Sum(h => h.StarvationDeaths),
                    herdingToolsConsumed = herds.Sum(h => h.HerdingToolsConsumed),
                    ownedAnimals = world.Herds.Where(h => h.DomesticCamp == id).Sum(h => h.Animals),
                    openingAnimals = receipt.OpeningAnimals, endingAnimals = receipt.EndingAnimals,
                    openingCapital = receipt.OpeningFieldCapital, endingCapital = receipt.EndingFieldCapital,
                    compiledStaffing = receipt.CompiledStaffing, production = receipt.Production, extraction = receipt.Extraction,
                    settledTrades = receipt.SettledTrades, settledGive = receipt.SettledGive, settledWant = receipt.SettledWant,
                    openingPopulation = receipt.OpeningPopulation, foodMet = receipt.FoodMet, waterMet = receipt.WaterMet,
                    moved = receipt.HasDecision && receipt.OpeningTile != receipt.EndingTile,
                    decisionDays = receipt.DecisionDays, endDays = receipt.EndDays,
                    elapsedYears = receipt.HasDecision ? (receipt.EndDays - receipt.DecisionDays) / world.Config.DaysPerYear : 0,
                    immigrants = receipt.Immigrated, emigrants = receipt.Emigrated, daughters = receipt.Daughters
                };
            });
        }

        internal static byte[] Derive(byte[] snapshot, int id, string transform)
        {
            var world = SaveCodec.Load(snapshot); var c = world.Camps.Single(x => x.Id == id);
            if (transform == "materials_empty") c.Inventory.Stone = c.Inventory.Clay = c.Inventory.Wood = 0;
            else if (transform == "tools_empty")
                c.Inventory.HandTools = c.Inventory.Tools = c.Inventory.FarmTools = c.Inventory.HerdingTools = c.Inventory.FishingTools = 0;
            else if (transform == "wood_three") c.Inventory.Wood = 3;
            else if (transform == "subsistence_needs_v13")
            {
                world.Config.Neolithic.SubsistenceNeedsCovered = true;
                foreach (var camp in world.Camps)
                    camp.PhysicalClothingPerCapita = SubsistenceNeeds.PhysicalClothingFraction(camp);
            }
            else if (transform != "none") throw new ArgumentException("Unknown analysis-only snapshot transformation.");
            return SaveCodec.Save(world);
        }
    }
}
