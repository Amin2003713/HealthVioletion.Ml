using Microsoft.ML.Data;

namespace HealthMulticlassClassification.Models;

public static class Const
{
    public static string Input { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Data" , "RestaurantScores.tsv");
    public static string Output { get; set; } = @"\Data\";
    public static string TrainedZip { get; set; } = @"\Model\";
}


public class Input
{
    [ColumnName("InspectionType") , LoadColumn(0)]
    public string InspectionType { get; set; }

    [ColumnName("ViolationDescription"), LoadColumn(1)]
    public string ViolationDescription { get; set; }

    [ColumnName("RiskCategory"), LoadColumn(2)]
    public string RiskCategory { get; set; }
}

public class Output
{
    [ColumnName("PredictedLabel")]
    public string Prediction { get; set; }
    public float[] Score { get; set; }
}