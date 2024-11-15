﻿using Microsoft.ML.Data;

namespace RecommendMovies.Models;

public class Input
{
    [LoadColumn(0)]
    public float UserId;

    [LoadColumn(1)]
    public float MovieId;

    [LoadColumn(2)]
    public float Label;
        
    [LoadColumn(3)]
    public float Timestamp;
}