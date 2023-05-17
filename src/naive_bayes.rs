use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug)]
pub struct StdNaiveBayes {
    pub alpha: f64,
    pub probability_of_class: HashMap<String, f64>,
    pub probability_of_feat_by_class: HashMap<String, HashMap<String, f64>>,
}

impl StdNaiveBayes {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            probability_of_class: HashMap::new(),
            probability_of_feat_by_class: HashMap::new(),
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<String>) {
        let mut y_counts: HashMap<String, i32> = HashMap::new();
        for class in y {
            let counter = y_counts.entry(class.to_string()).or_insert(0);
            *counter += 1;
        }

        let total_rows = y.len() as f64;
        let unique_classes: HashSet<String> = y.into_iter().cloned().collect();

        for uniq_class in &unique_classes {
            self.probability_of_class.insert(uniq_class.to_string(), *y_counts.get(uniq_class).unwrap() as f64 / total_rows);

            let mut class_feat_probs: HashMap<String, f64> = HashMap::new();
            let mut sum_of_feats_in_class = 0.0;
            for (i, class) in y.iter().enumerate() {
                if class == uniq_class {
                    for (j, feat_count) in x[i].iter().enumerate() {
                        let counter = class_feat_probs.entry(j.to_string()).or_insert(0.0);
                        *counter += *feat_count;
                        sum_of_feats_in_class += *feat_count;
                    }
                }
            }
            sum_of_feats_in_class += self.alpha * x[0].len() as f64;

            for (feat, count) in class_feat_probs.iter_mut() {
                *count = (*count + self.alpha) / sum_of_feats_in_class;
            }

            self.probability_of_feat_by_class.insert(uniq_class.to_string(), class_feat_probs);
        }
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<String> {
        let mut y_pred: Vec<String> = Vec::new();
        let unique_classes: Vec<String> = self.probability_of_class.keys().cloned().collect();
        let class_probabilities: Vec<f64> = self.probability_of_class.values().cloned().collect();
        let small_number = 1e-9;

        for row in x {
            let mut row_probabilities: Vec<f64> = Vec::new();
            for (i, class) in unique_classes.iter().enumerate() {
                let mut log_sum = (class_probabilities[i] + small_number).ln();
                for (j, feat_count) in row.iter().enumerate() {
                    if *feat_count > 0.0 {
                        let prob = self.probability_of_feat_by_class.get(class).unwrap().get(&j.to_string()).unwrap();
                        log_sum += (*feat_count * (*prob + small_number).ln());
                    }
                }
                row_probabilities.push(log_sum);
            }

            let max_prob_index = row_probabilities.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            y_pred.push(unique_classes[max_prob_index].to_string());
        }

        y_pred
    }
}
