using Microsoft.ML;
using Microsoft.ML.Trainers;
// ReSharper disable InconsistentNaming

namespace RecommendMovies.Trainer;

public static class ModelTrainer
{
    public static ITransformer BuildAndTrainMode(MLContext mlContext, IDataView trainingDataView)
    {
        var trainerEstimator = mlContext.Transforms.Conversion
            .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
            .Append(mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"))
            .Append(mlContext.Recommendation().Trainers.MatrixFactorization(
                new MatrixFactorizationTrainer.Options()
                {
                    MatrixColumnIndexColumnName = "userIdEncoded",
                    MatrixRowIndexColumnName = "movieIdEncoded",
                    LabelColumnName = "Label",
                    NumberOfIterations = 50,
                    ApproximationRank = 100,
                    NumberOfThreads = 3,
                }));
           

        return trainerEstimator.Fit(trainingDataView);
    }


    public static void Evaluate(MLContext mlContext, IDataView testDataView, ITransformer model)
    {
        Console.WriteLine("======================================");
        Console.WriteLine("🔍 Starting Model Evaluation");
        Console.WriteLine("======================================\n");

        // Generate predictions for the test data
        var predictions = model.Transform(testDataView);
        Console.WriteLine("📊 Predictions generated for test data.");

        // Evaluate the predictions using regression metrics
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
        Console.WriteLine("📈 Evaluation metrics calculated.\n");

        // Display evaluation results in a formatted table
        Console.WriteLine("=============== Evaluation Results ===============");
        Console.WriteLine($"| {"Metric",-25} | {"Value",-15} |");
        Console.WriteLine("-------------------------------------------------");
        Console.WriteLine($"| {"Root Mean Squared Error (RMSE)",-25} | {metrics.RootMeanSquaredError:N2,-15} |");
        Console.WriteLine($"| {"R Squared (R²)",-25}               | {metrics.RSquared:N2,-15} |");
        Console.WriteLine($"| {"Mean Absolute Error (MAE)",-25}    | {metrics.MeanAbsoluteError:N2,-15} |");
        Console.WriteLine($"| {"Loss Function",-25}                | {metrics.LossFunction:N2,-15} |");
        Console.WriteLine($"| {"Mean Squared Error (MSE)",-25}     | {metrics.MeanSquaredError:N2,-15} |");
        Console.WriteLine("=================================================\n");

        Console.WriteLine("✅ Model evaluation completed successfully.\n");
        Evaluate(mlContext, testDataView, model);
    }

}