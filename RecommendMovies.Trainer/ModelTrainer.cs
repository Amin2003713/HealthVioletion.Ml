using Microsoft.ML;
using Microsoft.ML.Trainers;
using RecommendMovies.Models;

// ReSharper disable InconsistentNaming

namespace RecommendMovies.Trainer;

public static class ModelTrainer
{


    public static void SaveModel(MLContext mlContext , ITransformer model , DataViewSchema schema)
    {
       mlContext.Model.Save(model , schema , Const.GetModelZipPath());
    }

    public static void UseModelSinglePrediction(MLContext mlContext, ITransformer model)
    {
        // Create a PredictionEngine to make a single prediction
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Input, Output>(model);
        Console.WriteLine(" Prediction engine created successfully.");

        // Define a sample input for testing
        var inputTest = new Input
        {
            UserId = 14,
            MovieId = 2683
        };

        Console.WriteLine(" Testing prediction for input:");
        Console.WriteLine($"   - UserId: {inputTest.UserId}");
        Console.WriteLine($"   - MovieId: {inputTest.MovieId}\n");

// Perform the prediction
        var prediction = predictionEngine.Predict(inputTest);

        Console.WriteLine("======================================");
        Console.WriteLine(" Prediction Result:");
        Console.WriteLine("======================================");
        Console.WriteLine($"   - Predicted Label (Rating): {prediction.Label}");
        Console.WriteLine($"   - Confidence Score: {prediction.Score:N2}");
        Console.WriteLine("======================================\n");

        Console.WriteLine(Math.Round(prediction.Score) >= 3.5F
            ? $"  {inputTest.MovieId} - Good choose! for user {inputTest.UserId}"
            : $"  {inputTest.MovieId} - Bad choose! for user {inputTest.UserId}");
    }

    public static ITransformer BuildAndTrainMode(MLContext mlContext, IDataView trainingDataView)
    {
        var trainerEstimator = mlContext.Transforms.Conversion
            .MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: "UserId")
            .Append(mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "MovieIdEncoded", inputColumnName: "MovieId"))
            .Append(mlContext.Recommendation().Trainers.MatrixFactorization(
                new MatrixFactorizationTrainer.Options()
                {
                    MatrixColumnIndexColumnName = "UserIdEncoded",
                    MatrixRowIndexColumnName = "MovieIdEncoded",
                    LabelColumnName = "Label",
                    NumberOfIterations = 1000,
                    ApproximationRank = 0,
                    NumberOfThreads = 4
                }));
           

        return trainerEstimator.Fit(trainingDataView);
    }


    public static void Evaluate(MLContext mlContext, IDataView testDataView, ITransformer model)
    {
        Console.WriteLine("======================================");
        Console.WriteLine("\t Starting Model Evaluation");
        Console.WriteLine("======================================\n");

        // Generate predictions for the test data
        var predictions = model.Transform(testDataView);
        Console.WriteLine("\t Predictions generated for test data.");

        // Evaluate the predictions using regression metrics
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
        Console.WriteLine("\t Evaluation metrics calculated.\n");

        // Display evaluation results in a formatted table
        Console.WriteLine("=============== Evaluation Results ===============");
        Console.WriteLine("| Metric | Value |");
        Console.WriteLine("-------------------------------------------------");
        Console.WriteLine($"| Rot Men Squared Error (RMSE) | {metrics.RootMeanSquaredError:##.####} |");
        Console.WriteLine($"| R Squared (R\u00b2)               | {metrics.RSquared:##.####} |");
        Console.WriteLine($"| Mean Absolute Error (MAE)    | {metrics.MeanAbsoluteError:##.####} |");
        Console.WriteLine($"| Loss Function                | {metrics.LossFunction:##.####} |");
        Console.WriteLine($"| Mean Squared Error (MSE)     | {metrics.MeanSquaredError:##.####} |");
        Console.WriteLine("=================================================\n");

        Console.WriteLine("\t Model evaluation completed successfully.\n");
    }

}