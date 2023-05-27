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
        // Adds words to the feature_names
        for sentence in input_document {
            for word in sentence.split(" ") {
                let word = word.to_string();
                if !self.feature_names.contains(&word) {
                    self.feature_names.push(word);
                }
            }
        }

        self.transform(input_document)
    }

    pub fn transform(&self, input_document: &Vec<String>) -> Vec<Vec<f64>> {
        let mut count_vector: Vec<Vec<f64>> = Vec::with_capacity(input_document.len());

        for (idx, sentence) in input_document.iter().enumerate() {
            count_vector.push(zeros(self.feature_names.len()));
            for word in sentence.split(" ") {
                let word = word.to_string();
                let position_of_word = self.feature_names.iter().position(|x| x == &word).unwrap();
                count_vector[idx][position_of_word] += 1f64;
            }
        }

        count_vector
    }
}

fn zeros(size: usize) -> Vec<f64> {
    let mut zero_vec: Vec<f64> = Vec::with_capacity(size);
    for i in 0..size {
        zero_vec.push(0.0);
    }
    return zero_vec;
}

#[cfg(test)]
mod feature_extraction_tests {
    use super::*;

    #[test]
    fn test_count_vector_fit_transform() {
        let mut count_vector = CountVectorizer::new();

        let document = vec![
            "hello this is ducky duck".to_string(),
            "chris don't wear wigs".to_string(),
            "ducks taste nice".to_string(),
            "duck duck goose".to_string(),
        ];

        let transformed_doc = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let feature_names = vec![
            "hello".to_string(),
            "this".to_string(),
            "is".to_string(),
            "ducky".to_string(),
            "duck".to_string(),
            "chris".to_string(),
            "don't".to_string(),
            "wear".to_string(),
            "wigs".to_string(),
            "ducks".to_string(),
            "taste".to_string(),
            "nice".to_string(),
            "goose".to_string(),
        ];

        assert_eq!(count_vector.fit_transform(&document), transformed_doc);
        assert_eq!(count_vector.feature_names, feature_names)
    }
}
