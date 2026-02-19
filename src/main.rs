use std::env;
use std::f32::consts::PI;

use hound;
use rustfft::{FftPlanner, num_complex::Complex};

const NOTE_NAMES: [&str; 12] = [
    "C","C#","D","D#","E","F",
    "F#","G","G#","A","A#","B"
];

// index 0 - root, index 4 - major third, index 7 - perfect fifth
// currently corresponds to C major
const MAJOR_TEMPLATE: [f32; 12] =
    [1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0];

// index 0 - root, index 3 - minor third, index 7 - perfect fifth
const MINOR_TEMPLATE: [f32; 12] =
    [1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0];

// converts note to pitch (eg. A3 -> A, D2 -> D etc.)
fn freq_to_pitch(freq: f32) -> usize {
    // TODO log2 as a lookup table for performance
    let midi = 69.0 + 12.0 * (freq / 440.0).log2();
    // round to nearest semitone
    let note_number = midi.round() as i32;
    // convert to pitch class (0–11)
    let mut pitch_class = note_number % 12;

    if pitch_class < 0 {
        pitch_class += 12;
    }

    pitch_class as usize
}

fn hann_window(n: usize) -> Vec<f32> {
    let mut window = vec![0.0; n];
    for i in 0..n {
        window[i] = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
    }
    window
}

// move the C major/minor template so that the root note corresponds to the chord played
fn roll_template(template: &[f32;12], shift: usize) -> [f32;12] {
    let mut out = [0.0;12];

    for i in 0..12 {
        out[(i + shift) % 12] = template[i];
    }

    out
}

fn dot(a: &[f32;12], b: &[f32;12]) -> f32 {
    let mut sum = 0.0;

    for i in 0..12 {
        sum += a[i] * b[i];
    }

    sum
}

fn find_chord(filename: &str) -> String {
    let mut reader = hound::WavReader::open(filename)
        .expect("Failed to open WAV");

    let spec = reader.spec();
    let sr = spec.sample_rate as usize;

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let n = samples.len();
    if n == 0 {
        return "Unknown".to_string();
    }

    // apply Hann window
    let window = hann_window(n);
    let mut input: Vec<Complex<f32>> = samples.iter()
        .zip(window.iter())
        .map(|(s,w)| Complex{ re: s*w, im: 0.0 })
        .collect();

    // FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut input);

    // build pitch-class energy vector
    let mut pitch_energy = [0.0f32; 12];

    for (k, c) in input.iter().enumerate().take(n/2) {
        let freq = k as f32 * sr as f32 / n as f32;

        if freq < 70.0 || freq > 1500.0 {
            continue;
        }

        let mag = c.norm();

        let pc = freq_to_pitch(freq);
        pitch_energy[pc] += mag;
    }

    // normalize
    if let Some(max_val) = pitch_energy.iter().cloned().reduce(f32::max) {
        if max_val > 0.0 {
            for v in pitch_energy.iter_mut() {
                *v /= max_val;
            }
        }
    }

    // get strongest 3 pitch classes
    let mut indices: Vec<usize> = (0..12).collect();
    indices.sort_by(|&a, &b| pitch_energy[a].partial_cmp(&pitch_energy[b]).unwrap());
    let top_notes: Vec<usize> = indices.iter().rev().take(3).cloned().collect();

    println!("Detected pitch classes:");
    for &i in &top_notes {
        println!("{}", NOTE_NAMES[i]);
    }

    // chord matching
    let mut best_score = 0.0;
    let mut best_chord = "Unknown".to_string();

    for root in 0..12 {
        let major = roll_template(&MAJOR_TEMPLATE, root);
        let minor = roll_template(&MINOR_TEMPLATE, root);

        let major_score = dot(&pitch_energy, &major);
        let minor_score = dot(&pitch_energy, &minor);

        if major_score > best_score {
            best_score = major_score;
            best_chord = format!("{} major", NOTE_NAMES[root]);
        }

        if minor_score > best_score {
            best_score = minor_score;
            best_chord = format!("{} minor", NOTE_NAMES[root]);
        }
    }

    if best_score < 0.4 {
        return "Unknown".to_string();
    }

    best_chord
}

fn main() {
    // TODO fix d major chord
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: cargo run -- <filename.wav>");
        return;
    }

    let chord = find_chord(&args[1]);
    println!("Detected chord: {}", chord);
}
