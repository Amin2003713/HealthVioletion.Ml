namespace ImageClassification.Model;

public class Output
{
    public string? ImagePath { get; set; }
    public string? Label { get; set; }
    public string? PredictedLabel { get; set; }
}


public class ImageData
{
    public string? ImagePath { get; set; }
    public string? Label { get; set; }

}