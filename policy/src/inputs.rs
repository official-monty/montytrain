use montyformat::chess::Position;
use montyvalue::input;

pub const INPUT_SIZE: usize = input::TOTAL;
pub const MAX_ACTIVE: usize = 128;

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Position, f: F) {
    input::map_features(pos, f);
}
