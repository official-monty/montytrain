use std::str::FromStr;

use bullet::{format::BulletFormat, inputs};
use montyformat::chess::{Castling, Position};

fn map_features<F: FnMut(usize)>(pos: &Position, mut f: F) {

}

#[derive(Clone, Copy)]
pub struct DataPoint {
    pub pos: Position,
    pub result: f32,
    pub score: i16,
}

impl FromStr for DataPoint {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            pos: Position::parse_fen(s, &mut Castling::default()),
            result: 0.5,
            score: 0,
        })
    }
}

impl BulletFormat for DataPoint {
    type FeatureType = usize;

    const HEADER_SIZE: usize = 0;

    fn set_result(&mut self, result: f32) {
        self.result = result;
    }

    fn result(&self) -> f32 {
        if self.pos.stm() == 0 { self.result } else { 1.0 - self.result }
    }

    fn result_idx(&self) -> usize {
        (2.0 * self.result()) as usize
    }

    fn score(&self) -> i16 {
        self.score
    }
}

impl IntoIterator for DataPoint {
    type IntoIter = ThreatInputsIter;
    type Item = (usize, usize);

    fn into_iter(self) -> Self::IntoIter {
        let mut res = ThreatInputsIter {
            features: [0; 128],
            active: 0,
            curr: 0,
        };

        map_features(&self.pos, |feat| {
            res.features[res.active] = feat;
            res.active += 1;
        });

        res
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;

pub struct ThreatInputsIter {
    features: [usize; 128],
    active: usize,
    curr: usize,
}

impl inputs::InputType for ThreatInputs {
    type RequiredDataType = DataPoint;
    type FeatureIter = ThreatInputsIter;

    fn buckets(&self) -> usize {
        1
    }

    fn max_active_inputs(&self) -> usize {
        32
    }

    fn inputs(&self) -> usize {
        768 * 4
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        pos.into_iter()
    }
}

impl Iterator for ThreatInputsIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr;

        if curr == self.active {
            None
        } else {
            self.curr += 1;
            let feat = self.features[curr];
            Some((feat, feat))
        }
    }
}
