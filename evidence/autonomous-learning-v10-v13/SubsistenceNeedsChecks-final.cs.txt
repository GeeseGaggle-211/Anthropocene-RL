using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using Anthropocene.Simulation;

internal static class SubsistenceNeedsChecks
{
    private static int checks, resolves, replay;
    private static void Check(bool value, string text) { checks++; if (!value) throw new Exception("Subsistence needs: " + text); }
    private static void Near(double value, double expected, string text) => Check(Math.Abs(value - expected) <= 1e-9 * Math.Max(1, Math.Abs(expected)), text + " actual=" + value + " expected=" + expected);
    private static WorldState Fixture(bool enabled = true, int other = 0, int explicitGatherers = 0)
    {
        var w = AutonomousElderChecks.Fixture(); var c = w.Camps[0];
        w.Config.Neolithic.SubsistenceNeedsCovered = enabled; w.Config.QoLMortalityEnabled = true; w.Config.AnnualHumanGrowthRate = .01145;
        w.Config.FissionEnabled = w.Config.ImmigrationEnabled = false;
        w.FixedHousing.Clear(); c.Inventory.HousingUnits = c.Inventory.LightHousingUnits = 0;
        c.Population = 100; c.StationaryYears = 100; c.ClothPerCapita = c.HousingPerCapita = 0; c.EconomyAutomationTurns = 0;
        c.QoL = .25; c.FoodNeedsMetRatio = c.WaterNeedsMetRatio = 1;
        c.Inventory.Cloth = c.Inventory.AnimalFabric = 0; c.Workforce = new PopulationLedger();
        if (other > 0) c.Workforce.Assignments.Add(new PopulationAssignment { Role = WorkforceRoles.Craftsperson, Population = other });
        if (explicitGatherers > 0) c.Workforce.Assignments.Add(new PopulationAssignment { Role = WorkforceRoles.Gatherer, Population = explicitGatherers });
        return w;
    }
    private static TurnRecord Resolve(WorldState w, List<Order> orders)
    {
        if (++resolves > 128) throw new Exception("Bounded needs proof exceeded128 resolves");
        return new TurnResolver().Resolve(w, TurnPlanner.Prepare(w, new Dictionary<int, List<Order>> { [0] = orders }));
    }
    private static string Magic(byte[] bytes)
    { using var zip = new GZipStream(new MemoryStream(bytes), CompressionMode.Decompress); using var reader = new BinaryReader(zip); return reader.ReadString(); }
    public static void Benchmark()
    {
        var samples = new List<object>(); int steps = 0;
        foreach (int n in new[] { 1, 80 }) foreach (bool enabled in new[] { false, true }) foreach (bool learned in new[] { false, true })
        {
            var times = new List<double>(); var allocations = new List<long>();
            for (int run = 0; run < 4; run++)
            {
                var w = AutonomousElderChecks.BenchmarkWorld(n); w.Config.Neolithic.SubsistenceNeedsCovered = enabled;
                AutonomousElderHistory.Initialize(w, w.Camps.Select(c => c.Id).ToArray());
                var e = new AutonomousElderHistoryEnvironment(growthAuthority: true);
                var o = e.Reset(SaveCodec.Save(w), w.Camps.Select(c => c.Id).ToArray());
                var actions = o.ToDictionary(p => p.Key, p => AutonomousElderCompiler.Uniform(p.Value));
                long start = GC.GetTotalAllocatedBytes(true); var timer = Stopwatch.StartNew();
                if (learned) e.Step(actions);
                else
                {
                    new TurnResolver().Resolve(w, TurnPlanner.Prepare(w, AiPlanning.Decide(w).Orders));
                    foreach (var camp in w.Camps.Where(c => c.Population > 0))
                    { var obs = AutonomousElderCompiler.Observe(w, camp); obs.SchemaVersion = 6; AutonomousElderHistoryEnvironment.Observe(w, camp.Id, obs); }
                    SaveCodec.Fingerprint(w);
                }
                timer.Stop(); steps++;
                if (run > 0) { times.Add(timer.Elapsed.TotalMilliseconds); allocations.Add(GC.GetTotalAllocatedBytes(false) - start); }
            }
            samples.Add(new { villages = n, enabled, mode = learned ? "autonomous6-uniform" : "heuristic",
                medianMs = times.OrderBy(v => v).ElementAt(1), minMs = times.Min(), maxMs = times.Max(), allocatedBytes = allocations.Average() });
        }
        Console.WriteLine(JsonSerializer.Serialize(new { version = "subsistence-needs-v13-complete-native-benchmark", steps, includes = "planning/economic execution/owner observations/fingerprint; excludes reset/setup/neural inference", samples }));
    }
    public static int Run()
    {
        checks = resolves = replay = 0;
        var w = Fixture(); var c = w.Camps[0]; var n = w.Config.Neolithic;
        NeolithicProduction.ResolveNeeds(w, c, 20);
        Near(c.ClothPerCapita, .8, "all Gatherers clothing baseline"); Near(c.HousingPerCapita, .8, "all Gatherers housing baseline");
        Near(c.FabricConsumed, 0, "automatic clothing consumes no fabric"); Near(c.PhysicalClothingPerCapita.Value, 0, "automatic coverage not physical fabric");
        Near(c.QoL, .8 * Math.Exp(-Math.Pow(100 / n.CohesionBandScale, n.SocialCohesionExponent)), "real zero shelter retains social cohesion");
        Near(NeolithicProduction.HousingCapacity(w, c, w.Days).Total, 0, "no virtual buildings");
        Near(NeolithicProduction.Demand(w, c, CraftGood.Cloth, 20), 40, "manufactured clothing can upgrade gear plus reserve");
        Near(NeolithicProduction.HousingGapPeople(w, c, 100, w.Days), 20, "remaining Gatherer housing need");
        var small = Fixture(); var sc = small.Camps[0]; sc.Population = 24; sc.StationaryYears = 0;
        NeolithicProduction.ResolveNeeds(small, sc, 20);
        Near(sc.QoL, .8 * NeolithicProduction.SocialCohesion(small, sc, 24), "early expectation cannot exceed gear ceiling");
        Check(sc.QoL >= .78 && sc.QoL <= .80 && QoLMortality.MaximumRate(small, sc) > 0, "well-fed small camp can regrow");
        double baselineGrowth = QoLMortality.MaximumRate(small, sc);
        sc.Inventory.Cloth = 4.8; NeolithicProduction.ResolveNeeds(small, sc, 20);
        Check(sc.QoL > .85 && QoLMortality.MaximumRate(small, sc) > baselineGrowth, "woven clothing raises QoL and growth above subsistence");
        sc.Inventory.AnimalFabric = 4.8; small.FixedHousing.Add(new FixedHousingStock { CampId = sc.Id, Tile = sc.Tile, Units = 1.2, LastUpdatedDays = small.Days });
        NeolithicProduction.ResolveNeeds(small, sc, 20);
        Check(sc.QoL > .95, "AnimalFabric plus real housing reaches higher quality");
        var mixed = Fixture(true, 60, 25); var m = mixed.Camps[0];
        Near(SubsistenceNeeds.CoveredResidents(mixed, m, 100), 32, "implicit plus explicit Gatherers baseline");
        NeolithicProduction.ResolveNeeds(mixed, m, 20);
        Near(m.ClothPerCapita, .32, "mixed empty clothing"); Near(m.HousingPerCapita, .32, "mixed empty housing");
        m.Inventory.Cloth = 80; mixed.FixedHousing.Add(new FixedHousingStock { CampId = m.Id, Tile = m.Tile, Units = 7.5, LastUpdatedDays = mixed.Days });
        NeolithicProduction.ResolveNeeds(mixed, m, 20);
        Near(m.FabricConsumed, 68, "60 nonGatherers plus Gatherer upgrades consume clothing"); Near(m.Inventory.Cloth, 12, "baseline need stock conserved");
        Near(m.ClothPerCapita, 1, "mixed full clothing"); Near(m.HousingPerCapita, .62, "32 automatic plus30 physically sheltered");
        Near(m.PhysicalClothingPerCapita.Value, .68, "physical clothing separately retained");
        Near(new Fission.QoLModel(mixed, m).At(100), m.QoL, "committed forecast matches effective needs");
        var incoming = new Fission.QoLModel(mixed, m);
        double incomingCoverage = .5 * (140d / 150) + .5 * (102d / 150); //72 automatic equivalents,68 physical clothes,30 real shelter
        Near(incoming.At(150), incomingCoverage * NeolithicProduction.SocialCohesion(mixed, m, 150), "immigrants become Gatherers and receive no virtual goods");
        var released = SubsistenceNeeds.ProjectWorkforce(m, 50);
        Near(released.Assignments.Where(a => a.Role != WorkforceRoles.Gatherer).Sum(a => a.Population), 30, "source retains proportionate jobs");
        var farmer = Fixture(true); farmer.Camps[0].Workforce.Assignments.Add(new PopulationAssignment { Role = WorkforceRoles.Farmer, Population = 100 });
        Near(SubsistenceNeeds.ProjectCovered(farmer, farmer.Camps[0], 50, true), 40, "daughter releases Farmer roles to Gatherers");
        foreach (bool enabled in new[] { false, true })
        {
            var full = Fixture(enabled, 100); var fc = full.Camps[0]; fc.Inventory.Cloth = 50;
            NeolithicProduction.ResolveNeeds(full, fc, 20); Near(fc.ClothPerCapita, .5, "nonGatherer clothing unchanged"); Near(fc.HousingPerCapita, 0, "nonGatherer housing unchanged");
        }
        var feed = Fixture(true, 60); var f = feed.Camps[0];
        var ff = NeolithicProduction.ClothingNeedFeedstock(n, SubsistenceNeeds.GoodsResidents(feed, f, 100), 20, 0, 0, 100, 0);
        Near(ff.Plant, 68, "clothing feedstock covers only accepted nonGatherers");
        var prepared = TurnPlanner.Prepare(feed, new Dictionary<int, List<Order>> { [0] = new List<Order> { new Order(OrderKind.Economy, f.Id) { Economy = EconomicIntent.From(f) } } });
        Check(prepared.Needs[f.Id].PlantClothingFeedstock <= 68, "preparation respects same clothing bound");
        var dispatch = Fixture(true); var dc = dispatch.Camps[0];
        var dp = TurnPlanner.Prepare(dispatch, new Dictionary<int, List<Order>> { [0] = new List<Order> { new Order(OrderKind.Dispatch, dc.Id, population: 20) } });
        Near(dp.Needs[dc.Id].Residents, 80, "dispatch locks resident goods cohort"); Near(dp.Needs[dc.Id].FoodConsumers, 100, "hunters still need food");
        Near(dp.Needs[dc.Id].PlantClothingFeedstock + dp.Needs[dc.Id].AnimalClothingFeedstock, 0, "dispatch/hunters require no manufactured clothing");
        var craft = Fixture(true, 1); var cc = craft.Camps[0]; cc.Inventory.HandTools = 100;
        craft.Config.Neolithic.WeaverFabricOutputPerBaseDays = 1; craft.Config.Neolithic.GeneralistTaskEfficiency = .25;
        var recipe = NeolithicProduction.Recipes(craft.Config).Single(r => r.Good == CraftGood.Cloth);
        double equipment = 100;
        var production = new TurnRecord();
        double used = NeolithicProduction.Produce(craft, cc, recipe, WorkforceRoles.Craftsperson, craft.Config.GenerationIntervals,
            craft.Config.GenerationIntervals, 1000, ref equipment, production);
        Near(production.Production.Single().Output, 20, "one general craftsperson makes20 cloth per20-year turn with inputs/tools");
        Near(used, 80, "production return is labor used, not output");
        Near(cc.Inventory.Cloth, 20, "actual cloth stock follows production receipt");
        var none = Fixture(true, 1); none.Camps[0].Inventory.HandTools = none.Camps[0].Inventory.Tools = 0;
        equipment = 0;
        Near(NeolithicProduction.Produce(none, none.Camps[0], recipe, WorkforceRoles.Craftsperson, none.Config.GenerationIntervals,
            none.Config.GenerationIntervals, 1000, ref equipment, new TurnRecord()), 0, "crafting still needs tools");
        var before = SaveCodec.Fingerprint(mixed); var bytes = SaveCodec.Save(mixed); var restored = SaveCodec.Load(bytes);
        Check(Magic(bytes) == "ANTHROPOCENE-14" && restored.Config.Neolithic.SubsistenceNeedsCovered, "enabled mode uses Save14");
        Check(before == SaveCodec.Fingerprint(restored), "Save14 exact fingerprint");
        Near(restored.Camps[0].PhysicalClothingPerCapita.Value, .68, "physical clothing fraction survives save");
        string enabledHash = SaveCodec.ConfigurationHash(mixed.Config);
        mixed.Config.Neolithic.SubsistenceNeedsCovered = false;
        Check(enabledHash != SaveCodec.ConfigurationHash(mixed.Config), "mode changes authoritative configuration hash");
        mixed.Config.Neolithic.SubsistenceNeedsCovered = true;
        var legacy = Fixture(false); var legacyBytes = SaveCodec.Save(legacy);
        Check(Magic(legacyBytes) == "ANTHROPOCENE-11" && !SaveCodec.Load(legacyBytes).Config.Neolithic.SubsistenceNeedsCovered, "legacy remains opt-out");
        legacy.Camps[0].PhysicalClothingPerCapita = .7;
        Check(legacyBytes.SequenceEqual(SaveCodec.Save(legacy)), "legacy bytes omit all new fields");
        var invalid = Fixture(true); invalid.Camps[0].PhysicalClothingPerCapita = double.NaN;
        bool rejected = false; try { SaveCodec.Save(invalid); } catch (InvalidDataException) { rejected = true; }
        Check(rejected, "malformed physical coverage rejected");
        var lag = Fixture(true); var lc = lag.Camps[0]; lc.Population = 24; double initial = lc.Population;
        Resolve(lag, new List<Order>()); Near(lc.Population, initial, "current need relief does not change prior-QoL growth plan");
        Check(lc.QoL > QoLMortality.ZeroGrowthQoL(lag.Config), "Gatherer needs permit next-turn growth");
        Resolve(lag, new List<Order>()); Check(lc.Population > initial, "next-turn food-rich population can recover");
        var results = new List<object>();
        foreach (bool scarce in new[] { false, true }) foreach (bool enabled in new[] { false, true })
        {
            var source = Fixture(enabled); var camp = source.Camps[0]; camp.Population = 24;
            if (scarce)
            {
                camp.Food = 0;
                foreach (var tile in source.Tiles) { tile.Food = tile.PotentialFood = tile.CultivationCapital = 0; for (int s = 0; s < 4; s++) tile.Seasons[s].MaxFood = tile.Seasons[s].MaxPotentialFood = 0; }
            }
            var env = new AutonomousElderHistoryEnvironment(growthAuthority: true); var obs = env.Reset(SaveCodec.Save(source), new[] { camp.Id }, 5);
            var trace = new List<object>(); double welfare = 0; string expectedResume = null;
            for (int i = 0; i < 5; i++)
            {
                var a = AutonomousElderCompiler.Uniform(obs[camp.Id]); Array.Clear(a.Workforce, 0, a.Workforce.Length); a.Workforce[0] = 1;
                Array.Clear(a.Activities[0], 0, a.Activities[0].Length); a.Activities[0][0] = 1;
                if (++resolves > 128) throw new Exception("Needs resolve budget");
                var r = env.Step(new Dictionary<int, AutonomousElderAction> { [camp.Id] = a }); obs = r.Observations;
                if (expectedResume != null) { Check(r.Fingerprint == expectedResume, "actual Save14 continuation matches uninterrupted next boundary"); expectedResume = null; }
                var state = SaveCodec.Load(env.CaptureSnapshot()); var now = state.Camps[0]; var demographic = state.History.Last().Demographics.Single(d => d.Camp == camp.Id);
                welfare += r.Rewards[camp.Id];
                trace.Add(new { population = now.Population, qol = now.QoL, cloth = now.ClothPerCapita, housing = now.HousingPerCapita,
                    food = now.FoodNeedsMetRatio, water = now.WaterNeedsMetRatio, physicalClothing = now.PhysicalClothingPerCapita,
                    fabricConsumed = now.FabricConsumed, births = demographic.Additions, deaths = demographic.Deaths,
                    growthCeiling = QoLMortality.MaximumRate(state, now) });
                Check(now.Inventory.Cloth == 0 && now.Inventory.AnimalFabric == 0 && NeolithicProduction.HousingCapacity(state, now, state.Days).Total == 0, "automatic coverage creates no physical assets");
                if (i == 1 && !r.Terminated)
                {
                    var resumed = new AutonomousElderHistoryEnvironment(growthAuthority: true);
                    var ro = resumed.Reset(env.CaptureSnapshot(), new[] { camp.Id }, 5);
                    var next = AutonomousElderCompiler.Uniform(ro[camp.Id]); Array.Clear(next.Workforce, 0, next.Workforce.Length); next.Workforce[0] = 1;
                    Array.Clear(next.Activities[0], 0, next.Activities[0].Length); next.Activities[0][0] = 1;
                    resolves++; var continued = resumed.Step(new Dictionary<int, AutonomousElderAction> { [camp.Id] = next });
                    expectedResume = continued.Fingerprint; replay++; AutonomousElderHistoryEnvironment.VerifyReplay(resumed.Replay); checks++;
                }
                if (r.Terminated || r.Truncated) break;
            }
            replay += env.Replay.Steps.Count; AutonomousElderHistoryEnvironment.VerifyReplay(env.Replay, new PhaseExecution(4, 1, true, 1)); checks++;
            var final = SaveCodec.Load(env.CaptureSnapshot()).Camps[0];
            if (scarce) Check(final.Population < 24, "food shortage still kills residents");
            else if (enabled) Check(final.Population > 24, "enabled food-rich Gatherers grow");
            else Near(final.Population, 24, "legacy food-rich Gatherers remain growth-suppressed");
            results.Add(new { enabled, scarce, welfare, population = final.Population, qol = final.QoL, trace });
        }
        Console.WriteLine(JsonSerializer.Serialize(new { version = "subsistence-needs-v13", result = "PASS", assertions = checks, resolves, replay, scenarios = results }));
        return checks;
    }
}
