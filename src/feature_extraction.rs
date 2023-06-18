use std::collections::HashSet;

/// Struct for converting a collection of text documents to a matrix of token counts.
/// This implementation produces a sparse representation of the counts using a Vector.
///
/// # Fields
/// `feature_names`: A vector storing the unique words found across all documents.
///     These are the 'features' that the model has learned.
///
/// # Examples
///
/// ```
/// use ducky_learn::feature_extraction::CountVectorizer;
///
/// let mut count_vector = CountVectorizer::new();
/// let document = vec![
///     "hello this is a test".to_string(),
///     "this is another test".to_string(),
/// ];
/// count_vector.fit_transform(&document);
/// assert_eq!(count_vector.feature_names, vec!["hello", "this", "is", "a", "test", "another"]);
/// ```
pub struct CountVectorizer {
    pub feature_names: Vec<String>,
}

impl CountVectorizer {
    /// Creates a new instance of `CountVectorizer` with an empty list of feature names.
    ///
    /// # Returns
    /// A new instance of `CountVectorizer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::feature_extraction::CountVectorizer;
    ///
    /// let count_vector = CountVectorizer::new();
    /// assert_eq!(count_vector.feature_names, Vec::<String>::new());
    /// ```
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    /// Fits the model according to the given training data and
    /// then transforms the data into a matrix of token counts.
    ///
    /// This process involves learning the 'vocabulary' from the input data (i.e.,
    /// all unique words across all documents) and then representing each document
    /// as a vector of counts of the words in the learned vocabulary.
    ///
    /// # Arguments
    /// * `input_document` - A vector of strings where each string represents a document.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector represents a document and contains
    /// the token counts for each word in the learned vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::feature_extraction::CountVectorizer;
    ///
    /// let mut count_vector = CountVectorizer::new();
    /// let document = vec![
    ///     "hello this is a test".to_string(),
    ///     "this is another test".to_string(),
    /// ];
    /// let transformed_document = count_vector.fit_transform(&document);
    /// assert_eq!(transformed_document, vec![
    ///     vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ///     vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    /// ]);
    /// ```
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

    /// Transforms the data into a matrix of token counts using the learned vocabulary.
    ///
    /// This process involves representing each document as a vector of counts of the
    /// words in the learned vocabulary. Note that this method does not learn the vocabulary
    /// and assumes that `fit_transform` has already been called.
    ///
    /// # Arguments
    /// * `input_document` - A vector of strings where each string represents a document.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector represents a document and contains
    /// the token counts for each word in the learned vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::feature_extraction::CountVectorizer;
    ///
    /// let mut count_vector = CountVectorizer::new();
    /// let document = vec![
    ///     "hello this is a test".to_string(),
    ///     "this is another test".to_string(),
    /// ];
    /// count_vector.fit_transform(&document);
    /// let new_document = vec![
    ///     "this another test".to_string(),
    /// ];
    /// let transformed_new_document = count_vector.transform(&new_document);
    /// assert_eq!(transformed_new_document, vec![
    ///     vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    /// ]);
    /// ```
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

/// Helper function that creates a new vector filled with zeros.
///
/// # Arguments
/// * `size` - The desired size of the vector.
///
/// # Returns
/// A new vector of the given size, filled with zeros.
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
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
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

    #[test]
    fn test_empty_string() {
        let mut count_vector = CountVectorizer::new();

        let document = vec!["".to_string()];

        let transformed_doc: Vec<Vec<f64>> = vec![vec![1.0]];

        let feature_names: Vec<String> = vec!["".to_string()];

        assert_eq!(count_vector.fit_transform(&document), transformed_doc);
        assert_eq!(count_vector.feature_names, feature_names)
    }
}
