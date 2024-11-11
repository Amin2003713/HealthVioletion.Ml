using Microsoft.ML;
using RecommendMovies.Models;
using RecommendMovies.Trainer;
using System;

var mlContext = new MLContext();
Console.WriteLine("======================================");
Console.WriteLine("🚀 Initialized MLContext");
Console.WriteLine("======================================\n");

Console.WriteLine("📂 Loading training data from path: " + Const.GetTrainPath());
var trainingDataView = mlContext
    .Data.LoadFromTextFile<Input>(path: Const.GetTrainPath(),
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true);
Console.WriteLine("✅ Training data loaded successfully.\n");

Console.WriteLine("📂 Loading test data from path: " + Const.GetTestPath());
var testDataView = mlContext
    .Data.LoadFromTextFile<Input>(path: Const.GetTestPath(),
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true);
Console.WriteLine("✅ Test data loaded successfully.\n");

Console.WriteLine("======================================");
Console.WriteLine("🛠️ Starting Model Training Process");
Console.WriteLine("======================================\n");

var model = ModelTrainer.BuildAndTrainMode(mlContext, trainingDataView);
Console.WriteLine("🎉 Model training completed.\n");

Console.WriteLine("======================================");
Console.WriteLine("📊 Evaluating Model with Test Data");
Console.WriteLine("======================================\n");

ModelTrainer.Evaluate(mlContext, testDataView, model);
Console.WriteLine("✅ Model evaluation completed.\n");
Console.WriteLine("======================================");
Console.WriteLine("🎉 All steps completed successfully!");
Console.WriteLine("======================================");