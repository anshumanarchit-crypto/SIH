# SpectraQ Real IQ Dataset Provenance

## Dataset Source

IDLab / imec – IQ samples of subGHz technologies

The dataset contains IQ captures from multiple Sub-GHz technologies,
including Sigfox, LoRa, IEEE 802.15.4g, IEEE 802.15.4 SUN-OFDM,
and IEEE 802.11ah.

Source:
https://idlab.ugent.be/resources/iq-samples-of-subghz-technologies

## Local File

Filename:
zigbee_g0.0dB_att22dB_freq867.4MHz_0.mat

Format:
MATLAB .mat

Variable:
IQ_samples

Data type:
double

Number of samples:
8,192,000

## Dataset Metadata

Documented sampling rate:
2.048 MHz

Centre frequency:
867.4 MHz

Capture method:
RTL-SDR using coaxial cables

## Protocol / Signal Identity

The filename contains "zigbee" and the dataset documentation identifies
IEEE 802.15.4g as one of the technologies included in the dataset.

The exact protocol identity of this individual local capture should not
be claimed beyond what is supported by the dataset documentation.

## SpectraQ Analysis

SpectraQ previously analysed the first 500,000 samples of this capture.

Measured results included:

- RMS magnitude: approximately 0.4993
- Peak magnitude: approximately 0.9301
- Dominant frequency offset: approximately 326.9 kHz
- Spectral centroid: approximately 338.0 kHz
- Spectral bandwidth: approximately 12.55 kHz
- 90% occupied bandwidth: approximately 31.35 kHz
- Activity ratio: approximately 0.30

These values are measurements produced by SpectraQ analysis and should
not be interpreted as independent protocol identification.

## Important Limitation

Absolute frequency measurements depend on the documented sampling rate
and centre-frequency metadata.

SpectraQ must not guess the absolute sampling frequency from the IQ
samples when reliable metadata is available.

## Licence

Licence information should be taken from the dataset's official
distribution/documentation before redistribution of the data.

## Source Verification Date

22 September 2026