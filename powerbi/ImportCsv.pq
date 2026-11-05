// Create a Text parameter named ExportFolder, e.g. C:\RetailRun\powerbi
// Create a Blank Query called LoadCsv, open Advanced Editor and paste this function.
// Import each table with a separate query: = LoadCsv("FactSales.csv")
(fileName as text) as table =>
let
    Separator = if Text.EndsWith(ExportFolder, "\") or Text.EndsWith(ExportFolder, "/") then "" else "\",
    Source = Csv.Document(File.Contents(ExportFolder & Separator & fileName), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    Headers
