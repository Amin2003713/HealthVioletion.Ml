using HealthMulticlassClassification.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace HealthMulticlassClassification.Trainer;

public static class BestModelFinder
{
    public static void FindBestModel()
    {
        Console.WriteLine("Starting best model finder ......");
        var mlContext = new MLContext();

        Console.WriteLine("Loading model ...");
        var trainingDataView = mlContext.Data.LoadFromTextFile<Input>(path: Const.Input , hasHeader: true, separatorChar: '\t' , allowQuoting: true);

        Console.WriteLine("Training PipeLine data ...  preprocessingPipeline ..");
        var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "RiskCategory", outputColumnName: "Prediction");

        Console.WriteLine("Mapping InputData .........");
        var inputMapper = preprocessingPipeline.Fit(trainingDataView).Transform(trainingDataView);

        Console.WriteLine("Multiclass Classification TrainingTime ......");
        Console.WriteLine("Training Setting ");
        var experimentSetting = new MulticlassExperimentSettings()
        {
            MaxExperimentTimeInSeconds = 300,
            CacheBeforeTrainer = CacheBeforeTrainer.On,
            OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
            CacheDirectoryName = null
        };
        var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSetting);

        Console.WriteLine("........................... result ......................");
        var experimentResult = experiment.Execute(trainData: inputMapper , labelColumnName: "RiskCategory" , progressHandler: new Progress<RunDetail<MulticlassClassificationMetrics>>());

        Console.WriteLine("............................... result Metrics ..........................");
        var metrics = experimentResult.BestRun.ValidationMetrics;
        Console.WriteLine($"Micro Accuracy is : {metrics.MicroAccuracy:N} ..........................");

    }
}