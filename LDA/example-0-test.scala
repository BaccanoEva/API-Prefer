// Stanford TMT Example 0 - Basic data loading
// http://nlp.stanford.edu/software/tmt/0.4/

import scalanlp.io._;

val pubmed = CSVFile("mashup.csv");

println("Success: " + pubmed + " contains " + pubmed.data.size + " records");

