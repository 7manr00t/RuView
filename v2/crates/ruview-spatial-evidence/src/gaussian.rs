//! `RfGaussian` / `GaussianMap` to `rf_gaussian` records.

use ruview_unified::gaussian::{GaussianMap, MotionState, RfGaussian};

use crate::wire::{
    sanitize_id, EvidenceError, EvidenceRecord, Motion, ProofTag, Provenance, RecordBody,
    RecordMeta, RfGaussianRecord,
};

/// Source id used when a Gaussian's device id sanitises to nothing.
const FALLBACK_SOURCE: &str = "ruview-unified";
/// Bounds for the envelope uncertainty derived from a Gaussian's scale.
const MIN_UNCERTAINTY_M: f64 = 1e-3;
const MAX_UNCERTAINTY_M: f64 = 100.0;

/// Caller context for exporting Gaussians.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianExport {
    /// Urth region id the room frame belongs to.
    pub region: String,
    /// Producer id written into provenance.
    pub producer: String,
    /// Strongest proof tag the caller claims. A Gaussian with synthetic
    /// RuView provenance is emitted as `SYNTHETIC` regardless.
    pub proof: ProofTag,
}

fn motion(m: MotionState) -> Motion {
    match m {
        MotionState::Static => Motion::Static,
        MotionState::Slow => Motion::Slow,
        MotionState::Fast => Motion::Fast,
    }
}

/// Envelope uncertainty for a Gaussian: the geometric mean of its three
/// σ, clamped to the v1 range. `RfGaussian` has no separate position
/// covariance, so its extent is the honest stand-in.
fn uncertainty(scale: [f64; 3]) -> f64 {
    let gm = (scale[0] * scale[1] * scale[2]).cbrt();
    gm.clamp(MIN_UNCERTAINTY_M, MAX_UNCERTAINTY_M)
}

/// Convert one Gaussian. `index` is its position in the map and makes the
/// receipt unique within one export.
pub fn gaussian_to_record(
    g: &RfGaussian,
    index: usize,
    ctx: &GaussianExport,
) -> Result<EvidenceRecord, EvidenceError> {
    let source_id = sanitize_id(&g.provenance.device_id, FALLBACK_SOURCE);
    let proof = if g.provenance.synthetic {
        ProofTag::Synthetic
    } else {
        ctx.proof
    };
    let receipt = sanitize_id(
        &format!("gauss:{source_id}:{}:{index}", g.timestamp_ns),
        "gauss",
    );
    let meta = RecordMeta {
        t_ns: g.timestamp_ns,
        region: ctx.region.clone(),
        source_id,
        uncertainty_m: uncertainty(g.scale),
        provenance: Provenance {
            receipt,
            producer: ctx.producer.clone(),
            proof,
        },
    };
    let body = RfGaussianRecord {
        position: g.position,
        scale: g.scale,
        orientation: g.orientation,
        occupancy: g.occupancy,
        confidence: g.confidence,
        motion: motion(g.motion),
        role: None,
    };
    EvidenceRecord::new(meta, RecordBody::RfGaussian(body))
}

/// Convert every Gaussian in a map, in the map's storage order. Fails on
/// the first Gaussian that does not satisfy the v1 rules.
pub fn map_to_records(
    map: &GaussianMap,
    ctx: &GaussianExport,
) -> Result<Vec<EvidenceRecord>, EvidenceError> {
    map.gaussians()
        .iter()
        .enumerate()
        .map(|(i, g)| gaussian_to_record(g, i, ctx))
        .collect()
}
