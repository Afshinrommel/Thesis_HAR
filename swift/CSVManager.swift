import Foundation

struct CSVManager {
    static func loadDataAndRunInference(fileName: String) {
        if let filepath = Bundle.main.path(forResource: fileName, ofType: "csv") {
            do {
                let fileContents = try String(contentsOfFile: filepath)
                print("Success: File loaded. Parsing data...")
                
                var parsedData = [[Double]]()
                let rows = fileContents.components(separatedBy: .newlines)
                
                for row in rows {
                    // Skip empty rows
                    guard !row.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                    
                    let columns = row.components(separatedBy: ",")
                    let doubleValues = columns.compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                    
                    // Ensures we only add rows that have valid data for your 13 columns (indices 0 to 12)
                    if doubleValues.count >= 13 {
                        parsedData.append(doubleValues)
                    }
                }
                
                print("Successfully parsed \(parsedData.count) rows. Starting inference...")
                
                // Initialize the verifier and pass the parsed data
                let verifier = NeuralInferenceVerifier()
                verifier.runBatchInference(fromCSV: parsedData)
                
            } catch {
                print("Error: Could not read the file contents.")
            }
        } else {
            print("Error: File not found in the project.")
        }
    }
}
