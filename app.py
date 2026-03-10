import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import math
import io
import warnings
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="MAE Reconstructor",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  GLOBAL CSS — warm light glassmorphism theme
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #0b1120 0%, #111827 40%, #0f172a 100%);
    color: #e2e8f0;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(14,165,233,0.15) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important; font-size: 11px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── headers ── */
h1, h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #f1f5f9 !important; }

/* ── glass card ── */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}

/* ── file uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed #475569 !important;
    border-radius: 16px !important;
    transition: all 0.3s ease;
    color: #cbd5e1 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0ea5e9 !important;
    background: rgba(14, 165, 233, 0.08) !important;
    box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.15) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #94a3b8 !important; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
}

/* ── download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
}

/* ── sliders ── */
.stSlider [data-baseweb="slider"] { padding: 4px 0; }
.stSlider [data-baseweb="thumb"] {
    background: #0ea5e9 !important;
    border: 3px solid #1e293b !important;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.4) !important;
    width: 20px !important; height: 20px !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg, #0ea5e9, #06b6d4) !important;
}

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important; font-size: 11px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 26px !important; font-weight: 600 !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(14, 165, 233, 0.15) !important;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

/* ── dividers ── */
hr { border-color: rgba(148, 163, 184, 0.15) !important; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #0ea5e9 !important; }

/* ── captions ── */
.stCaption {
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── image containers ── */
.img-panel {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
    border: 1px solid rgba(148, 163, 184, 0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.img-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* ── tag badges ── */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}
.tag-blue { background: rgba(29, 78, 216, 0.2); color: #60a5fa; }
.tag-amber { background: rgba(180, 83, 9, 0.2); color: #fbbf24; }
.tag-emerald { background: rgba(4, 120, 87, 0.2); color: #34d399; }
.tag-rose { background: rgba(190, 18, 60, 0.2); color: #fb7185; }

/* ── warnings & alerts ── */
[data-testid="stAlert"] {
    background: rgba(30, 41, 59, 0.7) !important;
    border-color: rgba(148, 163, 184, 0.2) !important;
    color: #cbd5e1 !important;
}

/* ── general text ── */
p, span, div { color: #cbd5e1; }
.stMarkdown { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  MODEL DEFINITION — matches checkpoint architecture
#  (timm-style ViT blocks, Conv2d patch embed,
#   encoder_to_decoder projection, pred_head output)
# ═══════════════════════════════════════════════

def random_masking(x, mask_ratio):
    """x: B x N x D — returns (visible, bool mask, ids_shuffle)"""
    B, N, D = x.shape
    num_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    x_visible = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :num_keep] = 0
    mask = mask.gather(1, ids_restore).bool()
    return x_visible, mask, ids_shuffle


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N, C


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_visible, keep_ids):
        """x_visible: B x num_keep x embed_dim (already embedded, not yet pos-encoded)"""
        B = x_visible.size(0)
        pos = self.pos_embed[:, 1:, :]
        x = x_visible + pos.expand(B, -1, -1).gather(
            1, keep_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:, :]  # drop cls token


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, patch_size=16, enc_dim=768, dec_dim=384, num_layers=12, num_heads=6):
        super().__init__()
        self.dec_dim = dec_dim
        self.num_patches = num_patches
        self.encoder_to_decoder = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        # +1 for cls token position (matches checkpoint shape [1, 197, 384])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dec_dim))
        self.blocks = nn.ModuleList([Block(dec_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred_head = nn.Linear(dec_dim, patch_size * patch_size * 3)

    def forward(self, enc_tokens, ids_shuffle, num_visible):
        B = enc_tokens.size(0)
        x = self.encoder_to_decoder(enc_tokens)
        mask_tokens = self.mask_token.expand(B, self.num_patches - num_visible, -1)
        full_seq = torch.cat([x, mask_tokens], dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        full_seq = full_seq.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_dim))
        # Use patch positions only (skip cls token at index 0)
        full_seq = full_seq + self.pos_embed[:, 1:, :]
        for blk in self.blocks:
            full_seq = blk(full_seq)
        full_seq = self.norm(full_seq)
        return self.pred_head(full_seq)


class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, enc_dim=768, dec_dim=384,
                 enc_layers=12, dec_layers=12, enc_heads=12, dec_heads=6, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder = MAEEncoder(img_size, patch_size, enc_dim, enc_layers, enc_heads)
        self.decoder = MAEDecoder(self.num_patches, patch_size, enc_dim, dec_dim, dec_layers, dec_heads)

    def forward(self, images):
        # Embed all patches with Conv2d patch_embed
        x = self.encoder.patch_embed(images)          # B x N x enc_dim
        # Random masking
        x_visible, mask, ids_shuffle = random_masking(x, self.mask_ratio)
        num_visible = x_visible.size(1)
        keep_ids = ids_shuffle[:, :num_visible]
        # Encode visible patches
        enc_out = self.encoder(x_visible, keep_ids)
        # Decode to full patch sequence
        reconstruction = self.decoder(enc_out, ids_shuffle, num_visible)
        # Return original patches for loss/display (B x N x patch_pixels)
        orig_patches = patchify(images)
        return reconstruction, mask, orig_patches


# ═══════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════

def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    h = w = H // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, patch_size * patch_size * C)


def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    h = w = img_size // patch_size
    x = patches.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, 3, h * patch_size, w * patch_size)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def build_masked_image(imgs, mask, patch_size=16):
    patches = patchify(imgs, patch_size)
    masked = patches.clone()
    for b in range(patches.shape[0]):
        masked[b][mask[b] == 1] = 0.5
    return unpatchify(masked, patch_size, imgs.shape[-1])


def tensor_to_pil(tensor):
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@st.cache_resource
def load_model(weights_path: str, device: str, mask_ratio: float):
    """Load MAE weights. Returns (model, error_msg). error_msg is None on success."""
    # Detect Git LFS pointer files (not real weights)
    try:
        with open(weights_path, 'rb') as f:
            header = f.read(8)
        if header[:7] == b'version':
            return None, "git-lfs"
    except Exception:
        pass
    model = MAE(mask_ratio=mask_ratio)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), None


# ═══════════════════════════════════════════════
#  SIDEBAR — dark navy panel
# ═══════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 0 28px 0;'>
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'>
            <div style='width:42px; height:42px; border-radius:12px;
                        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
                        display:flex; align-items:center; justify-content:center;
                        font-size:20px; box-shadow: 0 4px 12px rgba(14,165,233,0.4);'>
                🧩
            </div>
            <div>
                <div style='font-family:"Inter",sans-serif; font-size:18px;
                            font-weight:800; color:#f1f5f9; letter-spacing:-0.5px;'>
                    MAE Studio
                </div>
                <div style='font-family:"JetBrains Mono",monospace; font-size:9px;
                            color:#64748b; letter-spacing:1.5px; text-transform:uppercase;'>
                    Image Reconstructor
                </div>
            </div>
        </div>
        <div style='height:2px; background:linear-gradient(90deg, #0ea5e9, transparent);
                    border-radius:2px;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                color:#475569; letter-spacing:2px; text-transform:uppercase;
                margin-bottom:8px;'>
        Controls
    </div>
    """, unsafe_allow_html=True)

    mask_ratio = st.slider(
        "MASKING RATIO",
        min_value=0.1, max_value=0.95,
        value=0.75, step=0.05,
        help="Fraction of image patches hidden from the encoder"
    )

    n_visible = int(196 * (1 - mask_ratio))
    n_masked  = int(196 * mask_ratio)

    st.markdown(f"""
    <div style='background: rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.2);
                border-radius:12px; padding:16px; margin:16px 0;'>
        <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                    color:#64748b; letter-spacing:1.5px; text-transform:uppercase;
                    margin-bottom:12px;'>
            Patch Breakdown
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span style='color:#94a3b8; font-size:13px;'>Visible</span>
            <span style='color:#22d3ee; font-weight:700; font-family:"JetBrains Mono",monospace;
                         font-size:14px;'>{n_visible}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span style='color:#94a3b8; font-size:13px;'>Masked</span>
            <span style='color:#f59e0b; font-weight:700; font-family:"JetBrains Mono",monospace;
                         font-size:14px;'>{n_masked}</span>
        </div>
        <div style='height:1px; background:rgba(148,163,184,0.15); margin:8px 0;'></div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color:#94a3b8; font-size:13px;'>Total</span>
            <span style='color:#f1f5f9; font-weight:600; font-family:"JetBrains Mono",monospace;
                         font-size:14px;'>196</span>
        </div>
        <div style='margin-top:12px; height:6px; background:rgba(148,163,184,0.15);
                    border-radius:3px; overflow:hidden;'>
            <div style='height:100%; width:{100-mask_ratio*100:.0f}%;
                        background:linear-gradient(90deg, #22d3ee, #0ea5e9);
                        border-radius:3px;'></div>
        </div>
        <div style='font-family:"JetBrains Mono",monospace; font-size:9px;
                    color:#475569; margin-top:6px; text-align:right;'>
            {100-mask_ratio*100:.0f}% visible
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                color:#475569; letter-spacing:2px; text-transform:uppercase;
                margin-bottom:10px;'>
        Architecture
    </div>
    <table style='width:100%; font-size:12px; color:#94a3b8; border-spacing:0 6px;'>
        <tr><td>Encoder</td><td style='text-align:right; color:#22d3ee;'>ViT-Base</td></tr>
        <tr><td>Decoder</td><td style='text-align:right; color:#f59e0b;'>ViT-Small</td></tr>
        <tr><td>Patch Size</td><td style='text-align:right; color:#f1f5f9;'>16 x 16</td></tr>
        <tr><td>Image Size</td><td style='text-align:right; color:#f1f5f9;'>224 x 224</td></tr>
        <tr><td>Parameters</td><td style='text-align:right; color:#f1f5f9;'>~108M</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  HEADER BANNER — gradient hero
# ═══════════════════════════════════════════════

st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #164e63 100%);
            border-radius: 20px; padding: 40px 44px; margin-bottom: 28px;
            box-shadow: 0 8px 40px rgba(15,23,42,0.15);
            position: relative; overflow: hidden;'>
    <div style='position:absolute; top:-40px; right:-20px; width:180px; height:180px;
                border-radius:50%; background:rgba(14,165,233,0.12);'></div>
    <div style='position:absolute; bottom:-30px; right:80px; width:120px; height:120px;
                border-radius:50%; background:rgba(6,182,212,0.08);'></div>
    <div style='position:relative; z-index:1;'>
        <div style='display:flex; gap:8px; margin-bottom:14px;'>
            <span class='tag tag-blue'>Self-Supervised</span>
            <span class='tag tag-amber'>Vision Transformer</span>
        </div>
        <h1 style='font-size:clamp(26px,3.8vw,44px); font-weight:800; color:#f1f5f9;
                   line-height:1.15; margin:0 0 12px 0; letter-spacing:-1px;'>
            Masked Autoencoder<br>
            <span style='background: linear-gradient(90deg, #22d3ee, #0ea5e9);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                Image Reconstructor
            </span>
        </h1>
        <p style='color:#94a3b8; font-size:15px; max-width:520px; line-height:1.6; margin:0;'>
            Upload an image and watch the MAE reconstruct it from just
            <strong style='color:#22d3ee;'>{:.0f}%</strong> of visible patches,
            predicting the remaining
            <strong style='color:#f59e0b;'>{:.0f}%</strong> using learned representations.
        </p>
    </div>
</div>
""".format((1 - mask_ratio) * 100, mask_ratio * 100), unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  UPLOAD AREA
# ═══════════════════════════════════════════════

st.markdown("""
<div style='font-family:"Inter",sans-serif; font-size:13px; font-weight:600;
            color:#94a3b8; letter-spacing:0.5px; margin-bottom:8px;'>
    Upload your image
</div>
""", unsafe_allow_html=True)

upload_col, info_col = st.columns([2, 1])
with upload_col:
    uploaded = st.file_uploader(
        "Drop image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
with info_col:
    st.markdown("""
    <div class='glass-card' style='padding:16px 20px; text-align:center;'>
        <div style='font-size:28px; margin-bottom:8px;'>📐</div>
        <div style='font-family:"JetBrains Mono",monospace; font-size:11px;
                    color:#64748b; line-height:1.7;'>
            Accepts JPG, PNG, WEBP<br>
            Auto-resized to 224x224<br>
            Works on CPU & GPU
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  INFERENCE + RESULTS
# ═══════════════════════════════════════════════

if uploaded is not None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT = "model_mae.pth"
    model = None
    if os.path.exists(CHECKPOINT):
        try:
            result, err = load_model(CHECKPOINT, device, mask_ratio)
            if err == "git-lfs":
                model = MAE(mask_ratio=mask_ratio).to(device)
                model.eval()
                st.warning(
                    "**Demo mode** — `mae_best.pth` is a Git LFS pointer, not real weights. "
                    "Download the actual model file to see true reconstructions."
                )
            else:
                model = result
        except Exception as e:
            model = MAE(mask_ratio=mask_ratio).to(device)
            model.eval()
            st.warning(f"Could not load checkpoint ({e}) — running in **demo mode** with random weights.")
    else:
        model = MAE(mask_ratio=mask_ratio).to(device)
        model.eval()
        st.warning("Checkpoint not found — running with **random weights** (demo mode).")

    if model is not None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.open(uploaded).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        model.mask_ratio = mask_ratio

        with st.spinner("Reconstructing..."):
            with torch.no_grad():
                try:
                    recon_patches, mask, patches = model(img_tensor)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

        recon_imgs  = unpatchify(recon_patches, 16, 224)
        masked_imgs = build_masked_image(img_tensor.cpu(), mask.cpu(), 16)

        orig_np   = denormalize(img_tensor[0].cpu()).permute(1, 2, 0).numpy()
        masked_np = denormalize(masked_imgs[0].cpu()).permute(1, 2, 0).numpy()
        recon_np  = denormalize(recon_imgs[0].cpu()).permute(1, 2, 0).numpy()

        psnr_val = psnr_metric(orig_np, recon_np, data_range=1.0)
        ssim_val = ssim_metric(orig_np, recon_np, data_range=1.0, channel_axis=2)
        mse_val  = float(np.mean((orig_np - recon_np) ** 2))

        # ── Metrics Row ──
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("PSNR", f"{psnr_val:.2f} dB",
                   help="Peak Signal-to-Noise Ratio (higher = better)")
        mc2.metric("SSIM", f"{ssim_val:.4f}",
                   help="Structural Similarity (1.0 = perfect)")
        mc3.metric("MSE", f"{mse_val:.6f}",
                   help="Mean Squared Error (lower = better)")
        mc4.metric("Mask %", f"{mask_ratio*100:.0f}%",
                   help="Percentage of patches hidden")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Tabbed Results ──
        tab_results, tab_patchmap = st.tabs(["Results", "Patch Map"])

        with tab_results:
            c1, c2, c3 = st.columns(3, gap="medium")

            with c1:
                st.markdown("""
                <div class='img-panel'>
                    <div style='text-align:center; margin-bottom:8px;'>
                        <span class='tag tag-emerald'>Original</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(tensor_to_pil(torch.tensor(orig_np).permute(2, 0, 1)),
                         use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='img-panel'>
                    <div style='text-align:center; margin-bottom:8px;'>
                        <span class='tag tag-amber'>Masked ({mask_ratio*100:.0f}%)</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(tensor_to_pil(torch.tensor(masked_np).permute(2, 0, 1)),
                         use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div class='img-panel'>
                    <div style='text-align:center; margin-bottom:8px;'>
                        <span class='tag tag-blue'>Reconstruction</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(tensor_to_pil(torch.tensor(recon_np).permute(2, 0, 1)),
                         use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with tab_patchmap:
            st.markdown("""
            <div style='font-size:14px; font-weight:600; color:#f1f5f9; margin-bottom:12px;'>
                196 Patches &mdash; Visibility Grid (14 x 14)
            </div>
            """, unsafe_allow_html=True)

            mask_np = mask[0].cpu().numpy()

            grid_html = "<div style='display:inline-grid; grid-template-columns:repeat(14, 22px); gap:3px;'>"
            for i, m in enumerate(mask_np):
                if m == 1:
                    bg = "linear-gradient(135deg, #fbbf24, #f59e0b)"
                    border = "#f59e0b"
                    label = "masked"
                else:
                    bg = "linear-gradient(135deg, #22d3ee, #0ea5e9)"
                    border = "#0ea5e9"
                    label = "visible"
                grid_html += (
                    f"<div style='width:22px;height:22px;border-radius:4px;"
                    f"background:{bg};border:1px solid {border};"
                    f"box-shadow:0 1px 3px rgba(0,0,0,0.08);' "
                    f"title='Patch {i}: {label}'></div>"
                )
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)

            st.markdown("""
            <div style='margin-top:14px; display:flex; gap:24px; align-items:center;'>
                <div style='display:flex; align-items:center; gap:6px;'>
                    <div style='width:14px; height:14px; border-radius:3px;
                                background:linear-gradient(135deg, #22d3ee, #0ea5e9);'></div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:11px; color:#64748b;'>
                        Visible
                    </span>
                </div>
                <div style='display:flex; align-items:center; gap:6px;'>
                    <div style='width:14px; height:14px; border-radius:3px;
                                background:linear-gradient(135deg, #fbbf24, #f59e0b);'></div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:11px; color:#64748b;'>
                        Masked
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Download ──
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        dl_col, _ = st.columns([1, 2])
        with dl_col:
            recon_pil = tensor_to_pil(torch.tensor(recon_np).permute(2, 0, 1))
            buf = io.BytesIO()
            recon_pil.save(buf, format="PNG")
            st.download_button(
                label="Download Reconstruction",
                data=buf.getvalue(),
                file_name="mae_reconstruction.png",
                mime="image/png",
            )

else:
    # ── Empty state ──
    st.markdown("""
    <div class='glass-card' style='text-align:center; padding:60px 40px; margin-top:12px;'>
        <div style='font-size:56px; margin-bottom:20px;'>🧩</div>
        <div style='font-size:18px; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>
            Ready to reconstruct
        </div>
        <div style='color:#94a3b8; font-size:14px; max-width:400px; margin:0 auto 28px auto; line-height:1.6;'>
            Upload any image above. The model will mask patches, encode the visible ones,
            and reconstruct the full image.
        </div>
        <div style='display:flex; justify-content:center; gap:16px; flex-wrap:wrap;'>
            <div style='background:rgba(14,165,233,0.1); border:1px solid rgba(14,165,233,0.25); border-radius:12px;
                        padding:14px 20px; min-width:140px;'>
                <div style='font-size:20px; margin-bottom:4px;'>1.</div>
                <div style='font-size:12px; font-weight:600; color:#38bdf8;'>Upload Image</div>
            </div>
            <div style='background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); border-radius:12px;
                        padding:14px 20px; min-width:140px;'>
                <div style='font-size:20px; margin-bottom:4px;'>2.</div>
                <div style='font-size:12px; font-weight:600; color:#fbbf24;'>Set Mask Ratio</div>
            </div>
            <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); border-radius:12px;
                        padding:14px 20px; min-width:140px;'>
                <div style='font-size:20px; margin-bottom:4px;'>3.</div>
                <div style='font-size:12px; font-weight:600; color:#34d399;'>View Results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:20px 0; border-top:1px solid rgba(148,163,184,0.15);'>
    <div style='font-family:"JetBrains Mono",monospace; font-size:11px;
                color:#94a3b8; letter-spacing:0.5px; line-height:1.8;'>
        Masked Autoencoder &mdash; He et al., CVPR 2022<br>
        ViT-Base Encoder (768d, 12L) &bull; ViT-Small Decoder (384d, 12L)<br>
        Built with PyTorch & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
