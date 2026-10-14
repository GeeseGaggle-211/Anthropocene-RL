using System;
using System.Collections.Generic;
using System.Linq;
using Anthropocene.Simulation;

namespace Anthropocene.SnapshotBank
{
    internal sealed class SnapshotScenarioFactory
    {
        private readonly GenerationPlan plan;
        private readonly Dictionary<int, byte[]> baseWorlds =
            new Dictionary<int, byte[]>();

        public SnapshotScenarioFactory(GenerationPlan plan)
        {
            this.plan = plan ?? throw new ArgumentNullException(nameof(plan));
        }

        public ScenarioBuildResult Build(TrajectoryPlan trajectory)
        {
            var world = SaveCodec.Load(BaseWorld(trajectory.WorldSeed));
            int controlledPlayer = plan.BankKind == "cultivation-pilot" ||
                plan.BankKind == "elder-broad-domain" ||
                plan.BankKind == "elder-broad-domain-diagnostic" ? 1 : 0;
            var camp = world.Camps.Where(item => item.Player == controlledPlayer)
                .OrderBy(item => item.Id).FirstOrDefault()
                ?? throw new InvalidOperationException(
                    "Generated world has no requested controlled camp.");
            RemoveControlledArtifacts(world, camp);
            ConfigureCapabilities(world, camp, trajectory);
            int site = SelectSite(world, camp, trajectory,
                out string siteEvidence);
            camp.Tile = site;
            ConfigureCamp(world, camp, trajectory);
            ConfigureFields(world, camp, trajectory);
            ConfigureHerd(world, camp, trajectory);
            ConfigureFishing(world, camp, trajectory);
            ConfigureExternalDynamics(world, camp, trajectory);
            Ownership.Bind(world);
            FisheryState.Normalize(world);
            NeolithicProduction.NormalizeFixedHousing(world);
            TurnResolver.Invariants(world);
            byte[] canonical = SaveCodec.Save(world);
            world = SaveCodec.Load(canonical);
            if (!string.Equals(SaveCodec.Fingerprint(world),
                    SaveCodec.Fingerprint(SaveCodec.Load(
                        SaveCodec.Save(world))), StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Controlled fixture failed canonical SaveCodec roundtrip.");
            return new ScenarioBuildResult
            {
                World = world,
                ControlledCampId = camp.Id,
                SiteSelectionEvidence = siteEvidence
            };
        }

        private byte[] BaseWorld(int seed)
        {
            if (baseWorlds.TryGetValue(seed, out byte[] bytes))
                return bytes;
            var config = SimulationConfig.Small(plan.World.Players);
            config.Width = plan.World.Width;
            config.Height = plan.World.Height;
            config.Players = plan.World.Players;
            config.StartSeparation = plan.World.StartSeparation;
            config.AnnualHumanGrowthRate =
                plan.World.AnnualHumanGrowthRate;
            config.EconomicAllocation =
                EconomicAllocationStrategy.DemandControl;
            if (plan.BankKind == "cultivation-pilot" ||
                plan.BankKind == "elder-broad-domain" ||
                plan.BankKind == "elder-broad-domain-diagnostic" ||
                plan.BankKind == "autonomous-opportunity-v9")
            {
                config.TradeEnabled = true;
                config.ImmigrationEnabled = true;
                config.FissionEnabled = true;
                config.NeolithicFishingEnabled = plan.EnableHistoryFishing;
                config.CorrectedAquaticSeasonalPressureEnabled = plan.EnableHistoryFishing;
                config.FissionPressureRule =
                    FissionPressureRule.CurrentPlusExitOpportunity;
            }
            ValidateCapabilityThresholds(config);
            var generated = new WorldGenerator().Generate(seed, config);
            // GenerationReport contains wall-clock timings and process-memory
            // warnings. SaveCodec.Fingerprint deliberately excludes this
            // non-authoritative telemetry; canonicalize it here as well so
            // identical authoritative worlds also produce identical blobs.
            generated.Generation.Milliseconds = Enumerable.Repeat(0.0,
                generated.Generation.Stages.Count).ToList();
            generated.Generation.Warnings.Clear();
            bytes = SaveCodec.Save(generated);
            baseWorlds[seed] = bytes;
            return bytes;
        }

        private void ValidateCapabilityThresholds(SimulationConfig config)
        {
            string[] semanticKeys =
            {
                "field-retention",
                "herd-management",
                "resident-fishing"
            };
            if (plan.CapabilityExperienceThresholds.Count !=
                    semanticKeys.Length)
                throw new InvalidOperationException(
                    "The plan must declare exactly the native capability progression thresholds.");
            foreach (string key in semanticKeys)
            {
                if (!plan.CapabilityExperienceThresholds.TryGetValue(key,
                        out double threshold) ||
                    Math.Abs(threshold -
                        config.SecondaryTechnologyThreshold) > 1e-12)
                    throw new InvalidOperationException(
                        "Capability Experience threshold does not match native progression for " +
                        key + ".");
            }
        }

        private static void RemoveControlledArtifacts(WorldState world,
            Camp camp)
        {
            world.Hunters.RemoveAll(item => item.SourceCamp == camp.Id);
            world.Herds.RemoveAll(item => item.DomesticCamp == camp.Id);
            world.Improvements.RemoveAll(item =>
                item.Settlement == camp.Id);
            world.FixedHousing.RemoveAll(item => item.CampId == camp.Id);
            world.FisherySites.RemoveAll(item =>
                item.OperatorCampId == camp.Id);
            foreach (var tile in world.Tiles.Where(item =>
                         item.CultivationOperator == camp.Id))
            {
                tile.CultivationOperator = -1;
                tile.CultivationCapital = 0;
                tile.CultivationOperatingFraction = 0;
                tile.Cultivation = 0;
                tile.Cultivated = false;
            }
        }

        private static void ConfigureCapabilities(WorldState world, Camp camp,
            TrajectoryPlan trajectory)
        {
            bool cultivation = HasCultivation(trajectory.Economy);
            bool husbandry = HasHusbandry(trajectory.Economy);
            bool fishing = HasFishing(trajectory.Economy);
            var people = camp.People;
            people.Progression.Era = Era.Neolithic;
            people.Progression.CompletionDays = world.Days;
            people.Progression.CompletionTurn = world.Turn;
            people.Progression.CompletionRule = "controlled-valid-fixture";
            people.OpeningTransitionProcessed = true;
            people.OpeningTransitionProcessedInitialized = true;
            people.Technology = cultivation ? Technology.Cultivation :
                husbandry ? Technology.AnimalHusbandry :
                Technology.Watercraft;
            people.Technologies = new bool[4];
            people.Technologies[(int)people.Technology] = true;
            if (cultivation)
                people.Technologies[(int)Technology.Cultivation] = true;
            if (husbandry)
                people.Technologies[(int)Technology.AnimalHusbandry] = true;
            if (fishing)
                people.Technologies[(int)Technology.Watercraft] = true;
            people.Experience = new double[3];
            people.NeolithicExperience = new double[3];
            people.TechnologyExposure = new double[3];
            for (int domain = 0; domain < 3; domain++)
            {
                bool enabled = domain == (int)Domain.Gathering && cultivation ||
                    domain == (int)Domain.Hunting && husbandry ||
                    domain == (int)Domain.Fishing && fishing;
                people.Experience[domain] = enabled ?
                    world.Config.ExperienceThreshold * .8 : 0;
                people.NeolithicExperience[domain] = enabled ?
                    world.Config.SecondaryTechnologyThreshold * .35 : 0;
            }
            people.NeolithicTechnologies = new List<string>(
                NeolithicTech.FirstTier);
            people.CraftExperience = new Dictionary<string, double>();
            people.UnlockedSpecialists = new List<string>();
            people.FishingGroundKnowledge =
                new List<FishingGroundSnapshot>();
            people.Memory = new ContactMemory(world.Config.Width);
            if (string.Equals(trajectory.Frontier, "plough",
                    StringComparison.Ordinal))
                people.NeolithicTechnologies.Add(NeolithicTech.Plough);
            if (string.Equals(trajectory.Frontier,
                    "technology-transition-cultivation",
                    StringComparison.Ordinal))
            {
                people.Technologies[(int)Technology.Cultivation] = false;
                if (people.Technology == Technology.Cultivation)
                    people.Technology = Technology.AnimalHusbandry;
                people.NeolithicExperience[(int)Domain.Gathering] =
                    Math.Max(0, world.Config.SecondaryTechnologyThreshold -
                        1e-9);
            }
        }

        private static int SelectSite(WorldState world, Camp camp,
            TrajectoryPlan trajectory, out string evidence)
        {
            bool cultivation = HasCultivation(trajectory.Economy);
            var otherCamps = world.Camps.Where(item => item.Id != camp.Id &&
                item.Population > 0).ToArray();
            var candidates = new List<Tuple<int, double, int>>();
            foreach (var tile in world.Tiles)
            {
                if (!tile.TraversableLand || tile.Environment ==
                        EnvironmentKind.Ice)
                    continue;
                if (!(FreshwaterAccess.Available(world, tile.Id) > 0))
                    continue;
                if (otherCamps.Any(other => world.Map.Distance(tile.Id,
                        other.Tile) < world.Config.MinimumCampSpacing))
                    continue;
                if (cultivation && world.Map.Radius(tile.Id, 1).Count(id =>
                        CultivationEconomy.Cultivable(world.Tiles[id])) < 4)
                    continue;
                int fishing = VisibleFishingGrounds(world, tile.Id);
                if (trajectory.FishingGroundTarget == 1 && fishing != 1)
                    continue;
                if (trajectory.FishingGroundTarget == 2 && fishing != 2)
                    continue;
                if (trajectory.FishingGroundTarget == 3 && fishing < 3)
                    continue;
                if (trajectory.FishingGroundTarget > 0 &&
                    !world.Map.Neighbors(tile.Id).Any(id => IsFishingWater(
                        world.Tiles[id])))
                    continue;
                double support = PlacementStage.Viability(world, tile.Id);
                if (!(support > 0)) continue;
                candidates.Add(Tuple.Create(tile.Id, support, fishing));
            }
            if (candidates.Count == 0)
                throw new InvalidOperationException(
                    "No physically compatible site for requested fishing/field strata.");
            var ordered = candidates.OrderBy(item => item.Item2)
                .ThenBy(item => item.Item1).ToArray();
            int index = (int)Math.Round(trajectory.SiteSupportQuantile *
                (ordered.Length - 1), MidpointRounding.AwayFromZero);
            var selected = ordered[Math.Max(0, Math.Min(index,
                ordered.Length - 1))];
            evidence = "candidates=" + ordered.Length + ";rank=" + index +
                ";viability=" + selected.Item2.ToString("R",
                    System.Globalization.CultureInfo.InvariantCulture) +
                ";visible_fishing=" + Math.Min(3, selected.Item3);
            return selected.Item1;
        }

        private static void ConfigureCamp(WorldState world, Camp camp,
            TrajectoryPlan trajectory)
        {
            camp.Population = trajectory.Population;
            camp.PlantFood = camp.AnimalFood = camp.AquaticFood = 0;
            double years = trajectory.ReserveIntervals *
                world.Config.TurnDays / world.Config.DaysPerYear;
            double food = PhysicalFoodTimebase.StockForYears(world,
                camp.Population, years);
            bool cultivation = HasCultivation(trajectory.Economy);
            bool husbandry = HasHusbandry(trajectory.Economy);
            bool fishing = HasFishing(trajectory.Economy);
            if (fishing)
            {
                camp.AquaticFood = food * .65;
                camp.PlantFood = food * .25;
                camp.AnimalFood = food - camp.AquaticFood - camp.PlantFood;
            }
            else if (husbandry && !cultivation)
            {
                camp.AnimalFood = food * .60;
                camp.PlantFood = food - camp.AnimalFood;
            }
            else
            {
                camp.PlantFood = food * .85;
                camp.AnimalFood = food - camp.PlantFood;
            }
            camp.QoL = trajectory.AmenityCoverage < .5 ? .55 : 1;
            camp.FoodNeedsMetRatio = 1;
            camp.WaterNeedsMetRatio = 1;
            camp.ClothPerCapita = trajectory.AmenityCoverage;
            camp.HousingPerCapita = trajectory.AmenityCoverage;
            camp.DeficitTurns = 0;
            camp.FoodDeficit = camp.DeficitSeverity = 0;
            camp.LastConsumed = 0;
            camp.StationaryYears = 0;
            camp.StationaryTurns = 0;
            camp.IsVillage = cultivation;
            camp.VillageTurns = cultivation ? 1 : 0;
            camp.EconomyAutomationTurns = -1;
            camp.EconomicLabor = null;
            camp.Workforce = new PopulationLedger();
            camp.Inventory = new CampInventory();
            camp.CraftingPriorities = new CraftingPriorities();
            camp.ElderMemory = null;
            camp.LastSelectedFishingGrounds = new List<int>();
            camp.ManagedBreedingIntensity = .55;
            camp.ManagedBreedingPolicyInitialized = true;
            camp.HerdTargetSet = false;
            camp.HerdTarget = 0;
            double tools = trajectory.ToolAbundance * camp.Population;
            camp.Inventory.HandTools = tools;
            camp.Inventory.Tools = tools * .25;
            camp.Inventory.FarmTools = cultivation ? tools : 0;
            camp.Inventory.HerdingTools = husbandry ? tools : 0;
            camp.Inventory.FishingTools = fishing ? tools : 0;
            camp.Inventory.Pots = tools;
            camp.Inventory.Cloth = trajectory.AmenityCoverage *
                camp.Population;
            camp.Inventory.Stone = camp.Inventory.Clay =
                camp.Inventory.Wood = Math.Max(1, tools * 10);
            if (cultivation || fishing)
                NeolithicProduction.AddFixedHousing(world, camp,
                    trajectory.AmenityCoverage * camp.Population /
                    world.Config.Neolithic.HousingPeoplePerUnit,
                    world.Days);
            else
                camp.Inventory.LightHousingUnits =
                    trajectory.AmenityCoverage * camp.Population /
                    world.Config.Neolithic.LightHousingPeoplePerUnit;
            foreach (int id in world.Map.Radius(camp.Tile,
                         world.Config.VisionRadius))
                camp.People.Memory.Explored.Add(id, world.Turn);
        }

        private static void ConfigureFields(WorldState world, Camp camp,
            TrajectoryPlan trajectory)
        {
            if (!HasCultivation(trajectory.Economy)) return;
            double capital = Math.Pow(trajectory.FieldQ /
                world.Config.ReferenceIntensity, 2);
            foreach (int id in CultivationEconomy.Footprint(world, camp))
            {
                var tile = world.Tiles[id];
                tile.CultivationOperator = camp.Id;
                tile.CultivationCapital = capital;
                tile.CultivationOperatingFraction = 1;
                tile.Cultivation = (float)CultivationEconomy.Intensity(
                    world.Config, capital);
                tile.Cultivated = true;
            }
        }

        private static void ConfigureHerd(WorldState world, Camp camp,
            TrajectoryPlan trajectory)
        {
            if (!(trajectory.HerdSupportRatio > 0)) return;
            string species = world.Config.Species.Where(item =>
                    item.Domesticable).OrderBy(item => item.Id,
                    StringComparer.Ordinal).First().Id;
            var herd = new Herd
            {
                Id = world.NextId++,
                Tile = camp.Tile,
                DomesticCamp = camp.Id,
                OwnerSociety = camp.SocietyId,
                Species = species,
                Animals = 1,
                Capacity = world.Config.GetSpecies(species).MaximumHerdSize
            };
            world.Herds.Add(herd);
            var survey = LocalHerdEcology.Survey(world, camp.Tile, 1,
                world.Config.TurnDays);
            var support = LocalHerdEcology.Support(survey,
                HerdConsumptionProfile.Owned(world, camp),
                world.Config.GenerationIntervals);
            if (!(support.ZeroDisplacementCapacity > 0))
                throw new InvalidOperationException(
                    "Requested domestic herd has zero local sustainable support.");
            double requestedAnimals = support.ZeroDisplacementCapacity *
                trajectory.HerdSupportRatio;
            int animals = (int)Math.Round(requestedAnimals,
                MidpointRounding.AwayFromZero);
            int maximumHerdSize = world.Config.GetSpecies(species)
                .MaximumHerdSize;
            if (animals < 2)
                throw new InvalidOperationException(
                    "Requested domestic herd support ratio is below the minimum whole-animal fixture size.");
            if (animals > maximumHerdSize)
                throw new InvalidOperationException(
                    "Requested domestic herd support ratio exceeds the native maximum herd size.");
            herd.Animals = animals;
            herd.Capacity = maximumHerdSize;
            camp.HerdTargetSet = true;
            camp.HerdTarget = animals;
        }

        private static void ConfigureFishing(WorldState world, Camp camp,
            TrajectoryPlan trajectory)
        {
            if (!HasFishing(trajectory.Economy)) return;
            var access = FishingEconomy.BuildAccess(world, camp, camp.Tile,
                camp.Population, world.Days, world.Config.TurnDays);
            if (access.Grounds.Length < trajectory.FishingGroundTarget)
                throw new InvalidOperationException(
                    "Native Fishing access did not reach the requested number of grounds.");
            int[] selected = access.Grounds.Take(
                    trajectory.FishingGroundTarget)
                .Select(item => item.TileId).OrderBy(id => id).ToArray();
            double q = trajectory.FieldQ > 0 ? trajectory.FieldQ : .30;
            double capital = Math.Pow(q /
                world.Config.ReferenceIntensity, 2);
            foreach (int id in selected)
                world.FisherySites.Add(new FisherySite
                {
                    TileId = id,
                    OperatorCampId = camp.Id,
                    Capital = capital,
                    LastUpdatedDays = world.Days
                });
            camp.LastSelectedFishingGrounds = selected.ToList();
        }

        private static void ConfigureExternalDynamics(WorldState world,
            Camp controlled, TrajectoryPlan trajectory)
        {
            string variant = trajectory.CausalVariant ?? "none";
            if (trajectory.ScenarioFamily == "F16")
            {
                controlled.FissionPolicy = FissionPolicy.Neutral;
                controlled.StationaryTurns = 4;
                controlled.StationaryYears = 4 * world.Config.TurnDays /
                    world.Config.DaysPerYear;
                int radius = variant == "strong-exit-opportunity" ?
                    world.Config.FissionRadius : world.Config.VisionRadius;
                foreach (int tile in world.Map.Radius(controlled.Tile, radius))
                    controlled.People.Memory.Explored.Add(tile, world.Turn);
                return;
            }
            if (trajectory.ScenarioFamily != "F15")
                return;

            if (variant == "isolated")
            {
                world.FixedHousing.RemoveAll(item =>
                    item.CampId == controlled.Id);
                ConfigureCamp(world, controlled, VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, .02, 40.0, .85));
                return;
            }

            var partner = world.Camps.Where(item => item.Id != controlled.Id &&
                    item.Player != 0 && item.Player != controlled.Player)
                .OrderBy(item => item.Id).FirstOrDefault()
                ?? throw new InvalidOperationException(
                    "Open-economy fixture requires a non-player-zero partner camp.");
            RemoveControlledArtifacts(world, partner);
            partner.Tile = SelectPartnerSite(world, controlled, partner,
                preferRich: variant != "emigration-distress");

            TrajectoryPlan controlledProfile = trajectory;
            TrajectoryPlan partnerProfile;
            if (variant == "strong-food-import" ||
                variant == "food-shortfall-contact")
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, .02, 40.0, .85);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(100, trajectory.Population), .36, 4.0, .02, .15);
            }
            else if (variant == "mild-import")
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, .20, 1.5, .80);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(100, trajectory.Population), .36, 4.0, .05, .20);
            }
            else if (variant == "surplus-export")
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, 4.0, .02, .20);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(100, trajectory.Population), .32, .02, 1.8, .95);
            }
            else if (variant == "material-tool-import")
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, 4.0, .0, .15);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(100, trajectory.Population), .36, .10, 2.0, .95);
            }
            else if (variant == "emigration-distress")
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, .0, .0, .0);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(60, trajectory.Population / 3), .52, 4.0, 2.0, 1.0);
            }
            else if (variant == "immigration-stress")
            {
                controlledProfile = VariantPlan(trajectory,
                    Math.Max(70, trajectory.Population / 2), trajectory.FieldQ,
                    1.0, .5, .50);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(260, trajectory.Population), .16, .0, .0, .0);
            }
            else if (variant == "immigration-labor")
            {
                controlledProfile = VariantPlan(trajectory,
                    Math.Max(60, trajectory.Population / 2), trajectory.FieldQ,
                    4.0, 1.5, 1.0);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(260, trajectory.Population), .16, .0, .0, .0);
            }
            else
            {
                controlledProfile = VariantPlan(trajectory,
                    trajectory.Population, trajectory.FieldQ, 4.0, 1.5, 1.0);
                partnerProfile = VariantPlan(trajectory,
                    Math.Max(260, trajectory.Population), .16, .0, .0, .0);
            }

            world.FixedHousing.RemoveAll(item =>
                item.CampId == controlled.Id);
            ConfigureCamp(world, controlled, controlledProfile);
            ConfigureCapabilities(world, partner, partnerProfile);
            ConfigureCamp(world, partner, partnerProfile);
            ConfigureFields(world, partner, partnerProfile);
            ContactKnowledge.RecordTrade(world, controlled.People, partner);
            ContactKnowledge.RecordTrade(world, partner.People, controlled);
        }

        private static int SelectPartnerSite(WorldState world,
            Camp controlled, Camp partner, bool preferRich)
        {
            var candidates = world.Map.Radius(controlled.Tile,
                    world.Config.ImmigrationRadius)
                .Where(tile =>
                {
                    int distance = world.Map.Distance(controlled.Tile, tile);
                    if (distance < world.Config.MinimumCampSpacing ||
                        distance > world.Config.ImmigrationRadius)
                        return false;
                    var physical = world.Tiles[tile];
                    if (!physical.TraversableLand ||
                        !(FreshwaterAccess.Available(world, tile) > 0))
                        return false;
                    if (world.Map.Radius(tile, 1).Count(id =>
                            CultivationEconomy.Cultivable(world.Tiles[id])) < 4)
                        return false;
                    return world.Camps.Where(item => item.Id != partner.Id &&
                            item.Id != controlled.Id && item.Population > 0)
                        .All(item => world.Map.Distance(tile, item.Tile) >=
                            world.Config.MinimumCampSpacing);
                })
                .Select(tile => new
                {
                    Tile = tile,
                    Support = PlacementStage.Viability(world, tile)
                })
                .Where(item => item.Support > 0)
                .OrderBy(item => item.Support)
                .ThenBy(item => item.Tile)
                .ToArray();
            if (candidates.Length == 0)
                throw new InvalidOperationException(
                    "No physically compatible open-economy partner site.");
            return (preferRich ? candidates[candidates.Length - 1] :
                candidates[0]).Tile;
        }

        private static TrajectoryPlan VariantPlan(TrajectoryPlan source,
            int population, double fieldQ, double reserveIntervals,
            double toolAbundance, double amenityCoverage)
            => new TrajectoryPlan
            {
                TrajectoryId = source.TrajectoryId,
                ScenarioFamily = source.ScenarioFamily,
                ScenarioVersion = source.ScenarioVersion,
                WorldSeed = source.WorldSeed,
                TrajectoryFamily = source.TrajectoryFamily,
                LineageId = source.LineageId,
                CounterfactualGroupId = source.CounterfactualGroupId,
                SourceController = source.SourceController,
                Classification = source.Classification,
                Partition = source.Partition,
                HistoricalOrConsumedDiagnostic =
                    source.HistoricalOrConsumedDiagnostic,
                SnapshotSteps = source.SnapshotSteps,
                Economy = "cultivation",
                Population = population,
                FieldQ = fieldQ,
                HerdSupportRatio = 0,
                ReserveIntervals = reserveIntervals,
                FishingGroundTarget = 0,
                SiteSupportQuantile = source.SiteSupportQuantile,
                ToolAbundance = toolAbundance,
                AmenityCoverage = amenityCoverage,
                Frontier = "primitive",
                CausalVariant = source.CausalVariant,
                RequestedTags = source.RequestedTags
            };

        private static int VisibleFishingGrounds(WorldState world, int tile)
            => world.Map.Radius(tile, world.Config.VisionRadius)
                .Count(id => IsFishingWater(world.Tiles[id]));

        private static bool IsFishingWater(Tile tile)
            => tile.Environment == EnvironmentKind.Shallows ||
                tile.Environment == EnvironmentKind.Lake;

        internal static bool HasCultivation(string economy)
            => economy == "cultivation" || economy == "mixed" ||
                economy == "cultivation-fishing" ||
                economy == "all-three";

        internal static bool HasHusbandry(string economy)
            => economy == "husbandry" || economy == "mixed" ||
                economy == "husbandry-fishing" ||
                economy == "all-three" ||
                economy == "knowledge-only-husbandry";

        internal static bool HasFishing(string economy)
            => economy == "fishing" ||
                economy == "cultivation-fishing" ||
                economy == "husbandry-fishing" ||
                economy == "all-three";
    }
}
