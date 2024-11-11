using HealthMulticlassClassification.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Tensorflow;

namespace HealthMulticlassClassification.Trainer;

public static class ModelTrainer
{

    public static void TrainImbalanceDataset()
    {
    Console.WriteLine("Starting best model finder ......");
        var mlContext = new MLContext();

        Console.WriteLine("Loading model ...");
        var trainingDataView = mlContext.Data.LoadFromTextFile<Input>(Const.Input, hasHeader: true,
            separatorChar: '\t', allowQuoting: true);

        Console.WriteLine("Training PipeLine ...>  preprocessingPipeline ..");
        var preprocessingPipeline =
            mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "RiskCategory", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ViolationDescription",
                    outputColumnName: "Features"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);


        Console.WriteLine("Training Pipeline ....");
        var trainer = mlContext.MulticlassClassification
            .Trainers
            .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());


        Console.WriteLine(" Pipeline Setup ....");
        var trainingPipeLine = preprocessingPipeline
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        Console.WriteLine("cross validation model ....");
        var result = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeLine);


        Console.WriteLine("Accuracy of validation model ....");
        var macro = result.Average(a => a.Metrics.MacroAccuracy);
        var micro = result.Average(a => a.Metrics.MicroAccuracy);
        var logReduction = result.Average(a => a.Metrics.LogLossReduction);
        Console.WriteLine($"""
                           macro is : {macro}
                           Micro is : {micro}
                           LogLossReduction is : {logReduction}
                           """);

        Console.WriteLine("model fitting");
        var final = trainingPipeLine.Fit(trainingDataView);


        Console.WriteLine("Saving model ...");
        if (!Directory.Exists(Const.Output))
            Directory.CreateDirectory(Const.Output);

        mlContext.Model.Save(final, trainingDataView.Schema,
            Path.Combine(Const.Output, "ImbalanceDataset_RiskCategoryMulticlassClassificationModel.zip"));
        Console.WriteLine("Model saved to {0}", Const.Output);
    }


    public static void CreateImbalanceDataset(string input, string output)
    {
        var inputFileRow = File.ReadAllLines(input);

        if (inputFileRow.Length == 0)
            return;

        var outputFileRow = new List<string>();
        outputFileRow.add(inputFileRow.First());

        var highRiskCount = 0;
        var moderatRiskCount = 0;
        var lowRiskCount = 0;
        var random = new Random(0);

        foreach (var row in inputFileRow.Skip(1))
            if (row.Contains("High Risk"))
            {
                if (random.NextDouble() <= 1)
                {
                    highRiskCount++;
                    outputFileRow.Add(row);
                }
            }
            else if (row.Contains("Low Risk"))
            {
                if (random.NextDouble() <= 1)
                {
                    lowRiskCount++;
                    outputFileRow.Add(row);
                }
            }
            else if (row.Contains("Moderate Risk"))
            {
                if (random.NextDouble() <= 1)
                {
                    moderatRiskCount++;
                    outputFileRow.Add(row);
                }
            }

        File.WriteAllLines(output, outputFileRow);
        Console.WriteLine(outputFileRow.Count + "data valid retrieved From total " + inputFileRow.Length);
        Console.WriteLine(highRiskCount + "high risk found");
        Console.WriteLine(lowRiskCount + "low risk found");
        Console.WriteLine(moderatRiskCount + "moderate risk found");
    }


    public static void TrainAndSaveBestModel()
    {
        Console.WriteLine("Starting best model finder ......");
        var mlContext = new MLContext();

        Console.WriteLine("Loading model ...");
        var trainingDataView = mlContext.Data.LoadFromTextFile<Input>(Const.Input, hasHeader: true,
            separatorChar: '\t', allowQuoting: true);

        Console.WriteLine("Training PipeLine ...>  preprocessingPipeline ..");
        var preprocessingPipeline =
            mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "RiskCategory", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ViolationDescription",
                    outputColumnName: "Features"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);


        Console.WriteLine("Training Pipeline ....");
        var trainer = mlContext.MulticlassClassification
            .Trainers
            .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());


        Console.WriteLine(" Pipeline Setup ....");
        var trainingPipeLine = preprocessingPipeline
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        Console.WriteLine("cross validation model ....");
        var result = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeLine);


        Console.WriteLine("Accuracy of validation model ....");
        var macro = result.Average(a => a.Metrics.MacroAccuracy);
        var micro = result.Average(a => a.Metrics.MicroAccuracy);
        var logReduction = result.Average(a => a.Metrics.LogLossReduction);
        Console.WriteLine($"""
                           macro is : {macro}
                           Micro is : {micro}
                           LogLossReduction is : {logReduction}
                           """);

        Console.WriteLine("model fitting");
        var final = trainingPipeLine.Fit(trainingDataView);


        Console.WriteLine("Saving model ...");
        if (!Directory.Exists(Const.Output))
            Directory.CreateDirectory(Const.Output);

        mlContext.Model.Save(final, trainingDataView.Schema,
            Path.Combine(Const.Output, "RiskCategoryMulticlassClassificationModel.zip"));
        Console.WriteLine("Model saved to {0}", Const.Output);
    }


    public static void FindBestModel()
    {
        Console.WriteLine("Starting best model finder ......");
        var mlContext = new MLContext();

        Console.WriteLine("Loading model ...");
        var trainingDataView = mlContext.Data.LoadFromTextFile<Input>(Const.Input, hasHeader: true,
            separatorChar: '\t', allowQuoting: true);

        Console.WriteLine("Training PipeLine data ...  preprocessingPipeline ..");
        var preprocessingPipeline =
            mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "RiskCategory",
                outputColumnName: "Prediction");

        Console.WriteLine("Mapping InputData .........");
        var inputMapper = preprocessingPipeline.Fit(trainingDataView).Transform(trainingDataView);

        Console.WriteLine("Multiclass Classification TrainingTime ......");
        Console.WriteLine("Training Setting ");
        var experimentSetting = new MulticlassExperimentSettings
        {
            MaxExperimentTimeInSeconds = 300,
            CacheBeforeTrainer = CacheBeforeTrainer.On,
            OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
            CacheDirectoryName = null
        };
        var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSetting);

        Console.WriteLine("........................... result ......................");
        var experimentResult = experiment.Execute(inputMapper, "RiskCategory",
            progressHandler: new Progress<RunDetail<MulticlassClassificationMetrics>>());

        Console.WriteLine("............................... result Metrics ..........................");
        var metrics = experimentResult.BestRun.ValidationMetrics;
        Console.WriteLine($"Micro Accuracy is : {metrics.MicroAccuracy:N} ..........................");
        Console.WriteLine($"ConfusionMatrix is : {metrics.ConfusionMatrix} ..........................");
        Console.WriteLine($"LogLoss is : {metrics.LogLoss} ..........................");
        Console.WriteLine($"TopKAccuracyis : {metrics.TopKAccuracy} ..........................");
        Console.WriteLine($"PerClassLogLoss is : {metrics.PerClassLogLoss} ..........................");
        Console.WriteLine($"LogLossReduction is : {metrics.LogLossReduction} ..........................");
    }
}