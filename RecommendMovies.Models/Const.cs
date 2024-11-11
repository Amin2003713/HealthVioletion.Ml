namespace HealthMulticlassClassification.Models;

public static class Const
{
    public static string Input { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Data");
    public static string Output { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Model");
    public static string TrainedZip { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Data",
        "RestaurantScores.tsv");

    public static string Dataset { get; set; } = "RestaurantScores.tsv";
    public static string ImbalanceDataset { get; set; } = "ImbalanceDataset";
}