use std::collections::HashSet;

pub struct CountVectorizer {
    pub feature_names: Vec<String>,
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    pub fn fit_transform(&mut self, input_document: &Vec<String>) -> Vec<Vec<f64>> {
        for sentence in input_document {
            for word in sentence.split(" ") {
                if !self.feature_names.contains(&word) {
                    self.feature_names.push(word.to_string());
                }

            }
        }



        Vec::new()
    }
}