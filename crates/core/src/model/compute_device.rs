//! Chooses where tensors live: the processor or a graphics card.

use burn::tensor::Device;
use serde::{Deserialize, Serialize};

use crate::error::{BrainError, BrainResult};

/// Which compute backend to run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceChoice {
    /// The fastest backend this build includes.
    #[default]
    Automatic,
    /// Pure-Rust CPU backend; always available.
    Cpu,
    /// The processor through LibTorch, PyTorch's C++ engine (`libtorch`
    /// feature, on by default in the applications). About twice as fast as
    /// `Cpu` for training.
    Libtorch,
    /// Any graphics card, through Vulkan, Metal or DirectX 12 (`gpu` feature).
    Gpu,
    /// NVIDIA graphics cards through CUDA (`cuda` feature).
    Cuda,
}

impl DeviceChoice {
    /// Every choice, whether or not this build includes it.
    pub const ALL: [DeviceChoice; 5] = [
        DeviceChoice::Automatic,
        DeviceChoice::Cpu,
        DeviceChoice::Libtorch,
        DeviceChoice::Gpu,
        DeviceChoice::Cuda,
    ];

    /// The name used on the command line and in the user interface.
    pub fn name(self) -> &'static str {
        match self {
            DeviceChoice::Automatic => "auto",
            DeviceChoice::Cpu => "cpu",
            DeviceChoice::Libtorch => "libtorch",
            DeviceChoice::Gpu => "gpu",
            DeviceChoice::Cuda => "cuda",
        }
    }

    /// Looks a choice up by its name, ignoring letter case.
    pub fn from_name(name: &str) -> BrainResult<Self> {
        Self::ALL
            .into_iter()
            .find(|choice| choice.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| BrainError::ComputeDevice(format!("unknown device '{name}'")))
    }

    /// Whether this build was compiled with support for the choice.
    pub fn is_available(self) -> bool {
        match self {
            DeviceChoice::Automatic | DeviceChoice::Cpu => true,
            DeviceChoice::Libtorch => cfg!(feature = "libtorch"),
            DeviceChoice::Gpu => cfg!(feature = "gpu"),
            DeviceChoice::Cuda => cfg!(feature = "cuda"),
        }
    }

    /// The choices this build supports.
    pub fn available() -> Vec<DeviceChoice> {
        Self::ALL
            .into_iter()
            .filter(|choice| choice.is_available())
            .collect()
    }
}

/// The pair of devices a brain computes on.
#[derive(Debug, Clone)]
pub struct ComputeDevice {
    /// The backend that was actually selected.
    pub choice: DeviceChoice,
    /// Device for inference: no gradient bookkeeping.
    pub inference: Device,
    /// The same hardware with gradient tracking, for training.
    pub training: Device,
}

impl ComputeDevice {
    /// Selects a device.
    ///
    /// `Automatic` takes the fastest backend whose hardware is present: a
    /// graphics card, then LibTorch on the processor, then the pure-Rust
    /// processor backend. Any other choice fails when this build does not
    /// include the backend or the machine has no hardware for it.
    pub fn select(choice: DeviceChoice) -> BrainResult<Self> {
        let (choice, inference) = match choice {
            DeviceChoice::Automatic => select_fastest_working(),
            requested => (requested, create_device(requested)?),
        };
        Ok(Self {
            choice,
            training: inference.clone().autodiff(),
            inference,
        })
    }
}

/// Tries the accelerated backends this build includes, fastest first.
fn select_fastest_working() -> (DeviceChoice, Device) {
    [
        DeviceChoice::Cuda,
        DeviceChoice::Gpu,
        DeviceChoice::Libtorch,
    ]
    .into_iter()
    .filter(|choice| choice.is_available())
    .find_map(|choice| create_device(choice).ok().map(|device| (choice, device)))
    .unwrap_or((DeviceChoice::Cpu, Device::flex()))
}

fn create_device(choice: DeviceChoice) -> BrainResult<Device> {
    match choice {
        DeviceChoice::Cpu | DeviceChoice::Automatic => Ok(Device::flex()),
        DeviceChoice::Libtorch => create_libtorch_device(),
        DeviceChoice::Gpu => create_gpu_device(),
        DeviceChoice::Cuda => create_cuda_device(),
    }
}

#[cfg(feature = "libtorch")]
// Burn deprecates its LibTorch backend, but in 0.22.1 it trains about twice as
// fast on the processor as any non-deprecated backend (see the README).
#[allow(deprecated)]
fn create_libtorch_device() -> BrainResult<Device> {
    Ok(Device::libtorch())
}

#[cfg(not(feature = "libtorch"))]
fn create_libtorch_device() -> BrainResult<Device> {
    Err(missing_feature("libtorch"))
}

#[cfg(feature = "gpu")]
fn create_gpu_device() -> BrainResult<Device> {
    require_hardware(burn::tensor::DeviceType::Wgpu, "graphics card")?;
    Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice))
}

#[cfg(not(feature = "gpu"))]
fn create_gpu_device() -> BrainResult<Device> {
    Err(missing_feature("gpu"))
}

#[cfg(feature = "cuda")]
fn create_cuda_device() -> BrainResult<Device> {
    require_hardware(burn::tensor::DeviceType::Cuda, "CUDA graphics card")?;
    Ok(Device::cuda(burn::tensor::DeviceIndex::Default))
}

#[cfg(not(feature = "cuda"))]
fn create_cuda_device() -> BrainResult<Device> {
    Err(missing_feature("cuda"))
}

/// Fails with a clear message when the machine has no hardware for a backend.
///
/// Without this check the backend only discovers the problem on first use and
/// aborts the whole program.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn require_hardware(device_type: burn::tensor::DeviceType, hardware_name: &str) -> BrainResult<()> {
    let has_hardware = !Device::enumerate(device_type).into_vec().is_empty();
    if has_hardware {
        return Ok(());
    }
    Err(BrainError::ComputeDevice(format!(
        "no {hardware_name} was found on this machine; use the cpu device instead"
    )))
}

#[allow(dead_code)] // Unused only when every optional backend is compiled in.
fn missing_feature(feature_name: &str) -> BrainError {
    BrainError::ComputeDevice(format!(
        "this build does not include the '{feature_name}' backend; rebuild with `--features {feature_name}`"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_processor_backend_is_always_available() {
        let compute_device = ComputeDevice::select(DeviceChoice::Cpu).unwrap();

        assert_eq!(compute_device.choice, DeviceChoice::Cpu);
        assert!(compute_device.training.is_autodiff());
        assert!(!compute_device.inference.is_autodiff());
    }

    #[test]
    fn automatic_selection_always_finds_a_working_device() {
        let compute_device = ComputeDevice::select(DeviceChoice::Automatic).unwrap();

        assert_ne!(compute_device.choice, DeviceChoice::Automatic);
        assert!(compute_device.choice.is_available());
    }

    #[cfg(feature = "libtorch")]
    #[test]
    fn the_libtorch_backend_trains_when_built_in() {
        let compute_device = ComputeDevice::select(DeviceChoice::Libtorch).unwrap();
        let weights =
            burn::tensor::Tensor::<1>::from_data([1.0, 2.0, 3.0], &compute_device.training)
                .require_grad();
        let gradients = weights.clone().powf_scalar(2.0).sum().backward();

        let gradient =
            crate::model::tensor_readback::read_values(weights.grad(&gradients).unwrap());
        assert_eq!(gradient, vec![2.0, 4.0, 6.0]);
    }

    #[cfg(not(feature = "libtorch"))]
    #[test]
    fn a_missing_backend_explains_how_to_get_it() {
        let error = ComputeDevice::select(DeviceChoice::Libtorch).unwrap_err();

        assert!(error.to_string().contains("--features libtorch"));
    }

    #[test]
    fn device_names_round_trip() {
        for choice in DeviceChoice::ALL {
            assert_eq!(DeviceChoice::from_name(choice.name()).unwrap(), choice);
        }
    }
}
