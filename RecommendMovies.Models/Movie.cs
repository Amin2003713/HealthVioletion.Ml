namespace MovieRecommender;

internal class Movie
{
    public int movieId;

    public string MovieTitle;

   

    public Lazy<List<Movie>> _movies = new(() => LoadMovieData("moviesdatasetpath"));

    public Movie Get(int id)
    {
        return _movies.Value.Single(m => m.movieId == id);
    }

    private static List<Movie> LoadMovieData(string moviesdatasetpath)
    {
        var result = new List<Movie>();
        Stream fileReader = File.OpenRead(moviesdatasetpath);
        var reader = new StreamReader(fileReader);
        try
        {
            var header = true;
            var index = 0;
            var line = "";
            while (!reader.EndOfStream)
            {
                if (header)
                {
                    line = reader.ReadLine();
                    header = false;
                }

                line = reader.ReadLine();
                var fields = line.Split(',');
                var movieId = int.Parse(fields[0].TrimStart(new[] { '0' }));
                var movieTitle = fields[1];
                result.Add(new Movie { movieId = movieId, MovieTitle = movieTitle });
                index++;
            }
        }
        finally
        {
            reader?.Dispose();
        }

        return result;
    }
}