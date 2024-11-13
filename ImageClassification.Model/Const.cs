// ReSharper disable InconsistentNaming
namespace ImageClassification.Model;
public static class Const
{
    public static string Input { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Data");
    public static string Output { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Model");
    public static string TrainedZip { get; set; } = Path.Combine(
        new DirectoryInfo(Environment.CurrentDirectory).Parent!.Parent!.Parent!.FullName, "Data",
        "RestaurantScores.tsv");

    public static string Recommendation_Movies_Dataset { get; set; } = "recommendation-movies.csv";
    public static string Recommendation_Ratings_Test_Dataset { get; set; } = "recommendation-ratings-test.csv";
    public static string Recommendation_Ratings_Train_Dataset { get; set; } = "recommendation-ratings-train.csv";


    public static string GetTestPath() => Path.Combine(Input, Recommendation_Ratings_Test_Dataset);
    public static string GetTrainPath() => Path.Combine(Input, Recommendation_Ratings_Train_Dataset);
    public static string GetModelZipPath() => Path.Combine(Output, "Recommendation_Ratings_Train_Dataset.zip");
}