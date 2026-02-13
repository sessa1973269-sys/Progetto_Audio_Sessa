import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

audio_dir = os.path.join(base_dir, "..", "audio")
report_dir = os.path.join(base_dir, "..", "repot")

os.makedirs(audio_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# ================== PARAMETRI DI BASE ==================
print("=" * 80)
print("BEAMFORMING : DELAY-AND-SUM, MVDR E GSC")
print("=" * 80)

c = 343         # Velocità del suono [m/s]
d = 0.05        # Distanza tra i microfoni [m]
fs = 16000      # Frequenza di campionamento desiderata [Hz]
M = 4           # Numero di microfoni (aumentato a 4 per algoritmi avanzati)

# Angoli di arrivo (in radianti)
theta_target = 0           # Sorgente target "di fronte" (0 gradi)
theta_int = np.pi/3        # Sorgente interferente a 60 gradi

print(f"\nParametri:")
print(f"  - Velocità del suono: {c} m/s")
print(f"  - Distanza microfoni: {d} m")
print(f"  - Frequenza campionamento: {fs} Hz")
print(f"  - Numero microfoni: {M}")
print(f"  - Angolo target: {np.rad2deg(theta_target):.1f}°")
print(f"  - Angolo interferente: {np.rad2deg(theta_int):.1f}°")

# ================== FUNZIONI AUSILIARIE ==================

def apply_delay_samples(signal_in, delay_samples):
    """Applica un ritardo (positivo o negativo) al segnale."""
    L = len(signal_in)
    signal_out = np.zeros(L)
    
    if delay_samples >= 0:
        if delay_samples < L:
            signal_out[delay_samples:] = signal_in[:L-delay_samples]
    else:
        abs_delay = abs(delay_samples)
        if abs_delay < L:
            signal_out[:L-abs_delay] = signal_in[abs_delay:]
    
    return signal_out

def compute_snr(reference, signal_with_noise):
    """Calcola il rapporto segnale/rumore (SNR) in dB."""
    min_len = min(len(reference), len(signal_with_noise))
    ref = reference[:min_len]
    sig = signal_with_noise[:min_len]
    
    noise = sig - ref
    power_signal = np.mean(ref**2)
    power_noise = np.mean(noise**2)
    
    if power_noise < 1e-10:
        return float('inf')
    
    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db

def generate_test_signal(duration, fs, signal_type='target'):
    """Genera un segnale audio di test."""
    t = np.arange(0, duration, 1/fs)
    
    if signal_type == 'target':
        sig = (np.sin(2*np.pi*200*t) + 
               0.5*np.sin(2*np.pi*400*t) + 
               0.3*np.sin(2*np.pi*600*t))
        modulation = 0.5 + 0.5*np.sin(2*np.pi*3*t)
        sig = sig * modulation
    else:
        sig = (0.8*np.sin(2*np.pi*300*t) + 
               0.6*np.sin(2*np.pi*800*t) + 
               0.4*np.random.randn(len(t)))
    
    sig = sig / np.max(np.abs(sig)) * 0.5
    return sig

def compute_steering_vector(theta, freq, d, M, c):
    """
    Calcola il vettore di steering per un dato angolo e frequenza.
    
    Args:
        theta: angolo in radianti
        freq: frequenza in Hz
        d: distanza tra microfoni in m
        M: numero di microfoni
        c: velocità del suono in m/s
    
    Returns:
        steering vector (array complesso M x 1)
    """
    k = 2 * np.pi * freq / c  # numero d'onda
    a = np.zeros(M, dtype=complex)
    
    for m in range(M):
        x_m = m * d
        phase = k * x_m * np.sin(theta)
        a[m] = np.exp(-1j * phase)
    
    return a

def compute_beam_pattern(weights, freqs, angles, d, M, c):
    """
    Calcola il beam pattern (diagramma di direttività).
    
    Args:
        weights: pesi del beamformer (M x 1)
        freqs: frequenze da analizzare
        angles: angoli da -90° a 90° in radianti
        d, M, c: parametri geometrici
    
    Returns:
        pattern: matrice len(freqs) x len(angles) con guadagni in dB
    """
    pattern = np.zeros((len(freqs), len(angles)))
    
    for i, freq in enumerate(freqs):
        for j, theta in enumerate(angles):
            a = compute_steering_vector(theta, freq, d, M, c)
            # Risposta del beamformer
            response = np.abs(np.dot(weights.conj(), a))
            pattern[i, j] = 20 * np.log10(response + 1e-10)
    
    return pattern

def mvdr_beamformer(mic_signals, theta_look, freq_center, d, M, c, diagonal_loading=1e-3):
    """
    MVDR Beamformer (Minimum Variance Distortionless Response).
    
    Minimizza la varianza dell'uscita mantenendo guadagno unitario 
    nella direzione di look.
    
    Args:
        mic_signals: matrice L x M dei segnali ai microfoni
        theta_look: direzione di look in radianti
        freq_center: frequenza centrale per il design
        d, M, c: parametri geometrici
        diagonal_loading: regolarizzazione per stabilità numerica
    
    Returns:
        output: segnale in uscita
        weights: pesi ottimi del beamformer
    """
    L = len(mic_signals)
    
    # Calcola matrice di covarianza dei segnali
    Rxx = (mic_signals.T @ mic_signals) / L
    
    # Diagonal loading per stabilità
    Rxx = Rxx + diagonal_loading * np.eye(M)
    
    # Steering vector nella direzione di look
    a = compute_steering_vector(theta_look, freq_center, d, M, c)
    
    # Pesi MVDR: w = (R^-1 * a) / (a^H * R^-1 * a)
    try:
        Rxx_inv = np.linalg.inv(Rxx)
        numerator = Rxx_inv @ a
        denominator = np.conj(a) @ Rxx_inv @ a
        weights = numerator / denominator
    except np.linalg.LinAlgError:
        print("  ⚠️  Matrice singolare, uso pseudoinversa")
        Rxx_inv = np.linalg.pinv(Rxx)
        numerator = Rxx_inv @ a
        denominator = np.conj(a) @ Rxx_inv @ a
        weights = numerator / denominator
    
    # Applica i pesi ai segnali
    output = np.real(mic_signals @ weights)
    
    return output, weights

def gsc_beamformer(mic_signals, theta_look, freq_center, d, M, c, mu=0.01, iterations=1000):
    """
    GSC Beamformer (Generalized Sidelobe Canceller).
    
    Struttura a due rami:
    - Ramo superiore: beamformer fisso (delay-and-sum)
    - Ramo inferiore: filtro adattivo che cancella residui interferenti
    
    Args:
        mic_signals: matrice L x M
        theta_look: direzione target
        freq_center: frequenza centrale
        d, M, c: parametri geometrici
        mu: step size per LMS
        iterations: numero di iterazioni adattive
    
    Returns:
        output: segnale in uscita
        weights_adaptive: pesi adattivi finali
    """
    L = len(mic_signals)
    
    # RAMO SUPERIORE: Delay-and-Sum fisso verso theta_look
    aligned_upper = np.zeros((L, M))
    for m in range(M):
        x_m = m * d
        tau = (x_m * np.sin(theta_look)) / c
        D_samp = int(round(tau * fs))
        aligned_upper[:, m] = apply_delay_samples(mic_signals[:, m], -D_samp)
    
    y_upper = np.mean(aligned_upper, axis=1)
    
    # RAMO INFERIORE: Blocking Matrix + Filtro Adattivo
    # Blocking matrix: proietta su spazio ortogonale alla direzione di look
    # Semplificazione: usa differenze tra microfoni adiacenti
    B = np.zeros((M, M-1))
    for i in range(M-1):
        B[i, i] = 1
        B[i+1, i] = -1
    
    # Segnali bloccati (non contengono il target se è nella direzione giusta)
    x_blocked = mic_signals @ B
    
    # Filtro adattivo LMS
    weights_adaptive = np.zeros(M-1)
    y_adaptive = np.zeros(L)
    
    # Adattamento (elabora solo un sottoinsieme per velocità)
    step = max(1, L // iterations)
    for n in range(0, min(L, iterations * step), step):
        if n >= L:
            break
        # Uscita filtro adattivo
        y_adaptive[n] = np.dot(weights_adaptive, x_blocked[n, :])
        
        # Errore rispetto al ramo superiore
        error = y_upper[n] - y_adaptive[n]
        
        # Aggiornamento LMS: w(n+1) = w(n) + mu * e(n) * x(n)
        weights_adaptive += mu * error * x_blocked[n, :]
    
    # Applica filtro adattivo finale a tutti i campioni
    for n in range(L):
        y_adaptive[n] = np.dot(weights_adaptive, x_blocked[n, :])
    
    # Output finale: ramo superiore - ramo inferiore
    output = y_upper - y_adaptive
    
    return output, weights_adaptive

# ================== 

def normalize_audio(sig):
    """
    Normalizza il segnale e lo converte in int16
    per poterlo salvare correttamente in formato WAV.
    """
    max_val = np.max(np.abs(sig))
    
    if max_val < 1e-12:
        return np.int16(sig)
    
    sig_norm = sig / max_val
    return np.int16(sig_norm * 32767)

# ================== GENERAZIONE SEGNALI DI TEST ==================

print("\n" + "=" * 80)
print("GENERAZIONE SEGNALI SINTETICI ")
print("=" * 80)

duration = 3.0  # durata in secondi

# Genera sempre segnali sintetici
x_target = generate_test_signal(duration, fs, 'target')
x_int = generate_test_signal(duration, fs, 'interferer')

print("\n✓ Generati segnali sintetici:")
print("  - Target modulato (200, 400, 600 Hz)")
print("  - Interferente rumoroso (300, 800 Hz + rumore)")

# Allineamento lunghezze
L = min(len(x_target), len(x_int))
x_target = x_target[:L]
x_int = x_int[:L]

t = np.arange(L) / fs

print(f"\nSegnali pronti:")
print(f"  - Lunghezza: {L} campioni ({L/fs:.2f} secondi)")
print(f"  - Target: min={x_target.min():.3f}, max={x_target.max():.3f}")
print(f"  - Interferente: min={x_int.min():.3f}, max={x_int.max():.3f}")






# ================== SIMULAZIONE ARRAY DI MICROFONI ==================

print("\n" + "=" * 80)
print("SIMULAZIONE ARRAY DI MICROFONI")
print("=" * 80)

mic_signals = np.zeros((L, M))

# Contributo sorgente target
print(f"\nCalcolo ritardi per sorgente TARGET (θ={np.rad2deg(theta_target):.1f}°):")
for m in range(M):
    x_m = m * d
    tau_target = (x_m * np.sin(theta_target)) / c
    D_target = int(round(tau_target * fs))
    print(f"  Mic {m+1}: x={x_m:.3f}m, τ={tau_target*1e6:.2f}μs, delay={D_target} samples")
    mic_signals[:, m] += apply_delay_samples(x_target, D_target)

# Contributo sorgente interferente
print(f"\nCalcolo ritardi per sorgente INTERFERENTE (θ={np.rad2deg(theta_int):.1f}°):")
for m in range(M):
    x_m = m * d
    tau_int = (x_m * np.sin(theta_int)) / c
    D_int = int(round(tau_int * fs))
    print(f"  Mic {m+1}: x={x_m:.3f}m, τ={tau_int*1e6:.2f}μs, delay={D_int} samples")
    mic_signals[:, m] += apply_delay_samples(x_int, D_int)

mic1 = mic_signals[:, 0]

print(f"\n✓ Array di {M} microfoni simulato")

# ================== BEAMFORMING: CONFRONTO ALGORITMI ==================

print("\n" + "=" * 80)
print("APPLICAZIONE BEAMFORMING - CONFRONTO ALGORITMI")
print("=" * 80)

# 1) DELAY-AND-SUM
print("\n1) DELAY-AND-SUM Beamformer:")
aligned_ds = np.zeros((L, M))
for m in range(M):
    x_m = m * d
    tau_look = (x_m * np.sin(theta_target)) / c
    D_look = int(round(tau_look * fs))
    aligned_ds[:, m] = apply_delay_samples(mic_signals[:, m], -D_look)

y_ds = np.mean(aligned_ds, axis=1)
snr_ds = compute_snr(x_target, y_ds)
print(f"   SNR Delay-and-Sum: {snr_ds:.2f} dB")

# 2) MVDR BEAMFORMER
print("\n2) MVDR Beamformer:")
freq_center = 400  # Frequenza centrale per design (Hz)
y_mvdr, weights_mvdr = mvdr_beamformer(mic_signals, theta_target, freq_center, d, M, c)
snr_mvdr = compute_snr(x_target, y_mvdr)
print(f"   SNR MVDR: {snr_mvdr:.2f} dB")
print(f"   Pesi MVDR: {np.abs(weights_mvdr)}")

# 3) GSC BEAMFORMER
print("\n3) GSC Beamformer (adattivo):")
y_gsc, weights_gsc = gsc_beamformer(mic_signals, theta_target, freq_center, d, M, c, mu=0.001, iterations=500)
snr_gsc = compute_snr(x_target, y_gsc)
print(f"   SNR GSC: {snr_gsc:.2f} dB")
print(f"   Pesi adattivi finali: {weights_gsc}")

# SNR ingresso per confronto
snr_input = compute_snr(x_target, mic1)

# ================== BEAM PATTERN (DIAGRAMMA DI DIRETTIVITÀ) ==================

print("\n" + "=" * 80)
print("CALCOLO BEAM PATTERN")
print("=" * 80)

angles_deg = np.linspace(-90, 90, 181)
angles_rad = np.deg2rad(angles_deg)
freqs = [400, 800, 1600]  # Frequenze da analizzare

# Pesi per Delay-and-Sum (uniformi)
weights_ds = np.ones(M) / M

print("\nCalcolo pattern per Delay-and-Sum e MVDR...")
pattern_ds = compute_beam_pattern(weights_ds, freqs, angles_rad, d, M, c)
pattern_mvdr = compute_beam_pattern(weights_mvdr, freqs, angles_rad, d, M, c)

# ================== VALUTAZIONE E CONFRONTO ==================

print("\n" + "=" * 80)
print("RISULTATI FINALI")
print("=" * 80)

print(f"\n📊 CONFRONTO SNR:")
print(f"  {'Metodo':<20} {'SNR [dB]':<12} {'Miglioramento [dB]'}")
print(f"  {'-'*50}")
print(f"  {'Ingresso (Mic 1)':<20} {snr_input:>8.2f}     {'--'}")
print(f"  {'Delay-and-Sum':<20} {snr_ds:>8.2f}     {snr_ds - snr_input:>+8.2f}")
print(f"  {'MVDR':<20} {snr_mvdr:>8.2f}     {snr_mvdr - snr_input:>+8.2f}")
print(f"  {'GSC (Adattivo)':<20} {snr_gsc:>8.2f}     {snr_gsc - snr_input:>+8.2f}")

# --- 
# ================== SALVATAGGIO AUDIO ==================

from scipy.io.wavfile import write

write(os.path.join(audio_dir, "01_input_mic1.wav"), fs, normalize_audio(mic1))
write(os.path.join(audio_dir, "02_delay_and_sum.wav"), fs, normalize_audio(y_ds))
write(os.path.join(audio_dir, "03_mvdr.wav"), fs, normalize_audio(y_mvdr))
write(os.path.join(audio_dir, "04_gsc.wav"), fs, normalize_audio(y_gsc))

print("\n✓ File audio salvati nella cartella audio/")

# ================== VISUALIZZAZIONI ==================

print("\n" + "=" * 80)
print("GENERAZIONE GRAFICI")
print("=" * 80)

# FIGURA 1: Confronto forme d'onda
fig1, axes = plt.subplots(5, 1, figsize=(14, 12))

Nplot = min(3000, L)
t_plot = t[:Nplot]

axes[0].plot(t_plot, x_target[:Nplot], 'g', linewidth=1.5, label='Target')
axes[0].plot(t_plot, x_int[:Nplot], 'r', alpha=0.6, label='Interferente')
axes[0].set_ylabel('Ampiezza')
axes[0].set_title('Segnali Sorgente')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_plot, mic1[:Nplot], 'b')
axes[1].set_ylabel('Ampiezza')
axes[1].set_title(f'Microfono 1 (Ingresso) - SNR: {snr_input:.2f} dB')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_plot, y_ds[:Nplot], 'purple')
axes[2].set_ylabel('Ampiezza')
axes[2].set_title(f'Delay-and-Sum - SNR: {snr_ds:.2f} dB (Δ={snr_ds-snr_input:+.2f} dB)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_plot, y_mvdr[:Nplot], 'orange')
axes[3].set_ylabel('Ampiezza')
axes[3].set_title(f'MVDR - SNR: {snr_mvdr:.2f} dB (Δ={snr_mvdr-snr_input:+.2f} dB)')
axes[3].grid(True, alpha=0.3)

axes[4].plot(t_plot, y_gsc[:Nplot], 'cyan')
axes[4].set_xlabel('Tempo [s]')
axes[4].set_ylabel('Ampiezza')
axes[4].set_title(f'GSC (Adattivo) - SNR: {snr_gsc:.2f} dB (Δ={snr_gsc-snr_input:+.2f} dB)')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()

# FIGURA 2: Beam Pattern (Diagrammi di Direttività)
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

for i, freq in enumerate(freqs):
    axes2[0].plot(angles_deg, pattern_ds[i, :], label=f'{freq} Hz', linewidth=2)
    
axes2[0].axvline(np.rad2deg(theta_target), color='g', linestyle='--', label='Target', linewidth=2)
axes2[0].axvline(np.rad2deg(theta_int), color='r', linestyle='--', label='Interferente', linewidth=2)
axes2[0].set_xlabel('Angolo [gradi]')
axes2[0].set_ylabel('Guadagno [dB]')
axes2[0].set_title('Beam Pattern - Delay-and-Sum')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)
axes2[0].set_ylim([-40, 10])

for i, freq in enumerate(freqs):
    axes2[1].plot(angles_deg, pattern_mvdr[i, :], label=f'{freq} Hz', linewidth=2)
    
axes2[1].axvline(np.rad2deg(theta_target), color='g', linestyle='--', label='Target', linewidth=2)
axes2[1].axvline(np.rad2deg(theta_int), color='r', linestyle='--', label='Interferente', linewidth=2)
axes2[1].set_xlabel('Angolo [gradi]')
axes2[1].set_ylabel('Guadagno [dB]')
axes2[1].set_title('Beam Pattern - MVDR')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
axes2[1].set_ylim([-40, 10])

plt.tight_layout()

# FIGURA 3: Spettrogrammi
fig3, axes3 = plt.subplots(4, 1, figsize=(14, 10))

signals_to_plot = [
    (mic1, f'Microfono 1 - SNR: {snr_input:.2f} dB'),
    (y_ds, f'Delay-and-Sum - SNR: {snr_ds:.2f} dB'),
    (y_mvdr, f'MVDR - SNR: {snr_mvdr:.2f} dB'),
    (y_gsc, f'GSC - SNR: {snr_gsc:.2f} dB')
]

for idx, (sig, title) in enumerate(signals_to_plot):
    f, t_spec, Sxx = signal.spectrogram(sig, fs, nperseg=512)
    im = axes3[idx].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), 
                                shading='gouraud', cmap='viridis', vmin=-60, vmax=0)
    axes3[idx].set_ylabel('Freq [Hz]')
    axes3[idx].set_title(title)
    axes3[idx].set_ylim([0, 2000])
    if idx == 3:
        axes3[idx].set_xlabel('Tempo [s]')
    plt.colorbar(im, ax=axes3[idx], label='dB')

plt.tight_layout()
plt.show()
# ================== SALVATAGGIO GRAFICI ==================
fig1.savefig(os.path.join(report_dir, "01_waveforms.png"), dpi=300)
fig2.savefig(os.path.join(report_dir, "02_beam_pattern.png"), dpi=300)
fig3.savefig(os.path.join(report_dir, "03_spectrograms.png"), dpi=300)

print("\n✓ Grafici salvati nella cartella repot/")

# ================== ASCOLTO GUIDATO ==================

print("\n" + "=" * 80)
print("ASCOLTO GUIDATO - CONFRONTO ALGORITMI")
print("=" * 80)

# Normalizza segnali
signals_audio = {
    'Ingresso (Mic 1)': mic1 / np.max(np.abs(mic1)),
    'Delay-and-Sum': y_ds / np.max(np.abs(y_ds)),
    'MVDR': y_mvdr / np.max(np.abs(y_mvdr)),
    'GSC': y_gsc / np.max(np.abs(y_gsc))
}

for name, sig in signals_audio.items():
    print(f"\n🔊 Riproduzione: {name}")
    sd.play(sig, fs)
    sd.wait()
    input(f"   ✓ Completato. Premi INVIO per il prossimo...")

print("\n" + "=" * 80)
print("ELABORAZIONE COMPLETATA")
print("=" * 80)
print(f"\n🎯 RIEPILOGO:")
print(f"  • Array: {M} microfoni, spaziatura {d*100:.1f} cm")
print(f"  • Target: {np.rad2deg(theta_target):.0f}°, Interferente: {np.rad2deg(theta_int):.0f}°")
print(f"  • Algoritmo migliore: {'MVDR' if snr_mvdr > max(snr_ds, snr_gsc) else 'GSC' if snr_gsc > snr_ds else 'Delay-and-Sum'}")
print(f"  • Massimo miglioramento SNR: {max(snr_ds, snr_mvdr, snr_gsc) - snr_input:.2f} dB")
print("=" * 80)