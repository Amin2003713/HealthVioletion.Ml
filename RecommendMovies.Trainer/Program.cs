using Microsoft.ML;
using RecommendMovies.Models;
using RecommendMovies.Trainer;
using System;

var mlContext = new MLContext();
Console.WriteLine("======================================");
Console.WriteLine("\t Initialized MLContext");
Console.WriteLine("======================================\n");

Console.WriteLine("\t Loading training data from path: " + Const.GetTrainPath());
var trainingDataView = mlContext
    .Data.LoadFromTextFile<Input>(path: Const.GetTrainPath(),
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true);
Console.WriteLine("\t Training data loaded successfully.\n");

Console.WriteLine("\t Loading test data from path: " + Const.GetTestPath());
var testDataView = mlContext
    .Data.LoadFromTextFile<Input>(path: Const.GetTestPath(),
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true);
Console.WriteLine("\t Test data loaded successfully.\n");

Console.WriteLine("======================================");
Console.WriteLine("\t Starting Model Training Process");
Console.WriteLine("======================================\n");

var model = ModelTrainer.BuildAndTrainMode(mlContext, trainingDataView);
Console.WriteLine("\t Model training completed.\n");

Console.WriteLine("======================================");
Console.WriteLine("\t Evaluating Model with Test Data");
Console.WriteLine("======================================\n");

ModelTrainer.Evaluate(mlContext, testDataView, model);
Console.WriteLine("\t Model evaluation completed.\n");
Console.WriteLine("======================================");
Console.WriteLine("\t All steps completed successfully!");
Console.WriteLine("======================================");

 ModelTrainer.UseModelSinglePrediction(mlContext, model);


 ModelTrainer.SaveModel(mlContext , model , trainingDataView.Schema);







Console.ReadKey();