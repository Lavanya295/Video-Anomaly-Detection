import os, cv2, numpy as np, matplotlib, warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio

BASE = r"c:\Users\atanu\Desktop\TEST"
VID_DIR = os.path.join(BASE, "videos")
GIF_DIR = os.path.join(BASE, "gifs")
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

H, W, FPS, N = 240, 320, 10, 60

PALETTE = {
    "bg":"#0d1117","panel":"#161b22",
    "accent1":"#58a6ff","accent2":"#3fb950","accent3":"#f78166",
    "text":"#e6edf3","subtext":"#8b949e",
}

def write_video(path, frames):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H))
    for f in frames: out.write(f)
    out.release()

def make_normal():
    frames, base = [], np.full((H,W,3),40,dtype=np.uint8)
    for i in range(N):
        f=base.copy(); cx=int(W*0.2+(W*0.6)*(i/N))
        cv2.circle(f,(cx,H//2),18,(120,180,80),-1)
        f=cv2.add(f,np.random.randint(0,8,(H,W,3),dtype=np.uint8))
        frames.append(f)
    return frames

def make_crowd():
    frames, base = [], np.full((H,W,3),40,dtype=np.uint8)
    for i in range(N):
        f=base.copy(); cx=int(W*0.2+(W*0.6)*(i/N))
        cv2.circle(f,(cx,H//2),18,(120,180,80),-1)
        f=cv2.add(f,np.random.randint(0,8,(H,W,3),dtype=np.uint8))
        if 34<=i<=45:
            for _ in range(30):
                cv2.circle(f,(np.random.randint(60,W-60),np.random.randint(60,H-60)),np.random.randint(5,18),(0,60,200),-1)
        frames.append(f)
    return frames

def make_object():
    frames, base = [], np.full((H,W,3),40,dtype=np.uint8)
    for i in range(N):
        f=base.copy(); cx=int(W*0.2+(W*0.6)*(i/N))
        cv2.circle(f,(cx,H//2),18,(120,180,80),-1)
        f=cv2.add(f,np.random.randint(0,8,(H,W,3),dtype=np.uint8))
        if 19<=i<=35:
            dy=int(H*0.25+(i-19)*4)
            cv2.rectangle(f,(W//2-30,dy),(W//2+30,dy+55),(0,0,220),-1)
        frames.append(f)
    return frames

def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1]+(4,))
    return buf[:,:,:3]

def frame_diff_scores(frames):
    s=[0.0]
    for i in range(1,len(frames)):
        p=cv2.cvtColor(frames[i-1],cv2.COLOR_BGR2GRAY).astype(float)
        c=cv2.cvtColor(frames[i],  cv2.COLOR_BGR2GRAY).astype(float)
        s.append(np.abs(c-p).mean())
    return np.array(s)

print("Generating videos...")
nf=make_normal(); cf=make_crowd(); of=make_object()
write_video(os.path.join(VID_DIR,"normal_traffic.avi"),nf)
write_video(os.path.join(VID_DIR,"anomaly_crowd.avi"),cf)
write_video(os.path.join(VID_DIR,"anomaly_object.avi"),of)
print("Videos saved.")

ns=frame_diff_scores(nf); cs=frame_diff_scores(cf); os_=frame_diff_scores(of)
gt_c=np.zeros(N); gt_c[34:46]=1
gt_o=np.zeros(N); gt_o[19:36]=1
all_s=np.concatenate([ns,cs,os_])
all_gt=np.concatenate([np.zeros(N),gt_c,gt_o])
mu,sig=ns.mean(),ns.std()
THR=mu+2.5*sig
print(f"Threshold={THR:.2f}")

# GIF 01 Pipeline
print("GIF 01...")
STEPS=[("Video\nIngest","#1f6feb"),("Frame\nExtract","#388bfd"),("Frame\nDiff","#58a6ff"),("Z-Score","#79c0ff"),("Anomaly\nFlag","#f78166")]
gf=[]
for si in range(len(STEPS)):
    fig,ax=plt.subplots(figsize=(10,3.2),facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.set_xlim(-0.5,len(STEPS)-0.5); ax.set_ylim(-0.8,1.2); ax.axis("off")
    ax.set_title("Video Anomaly Detection — Pipeline",color=PALETTE["text"],fontsize=14,fontweight="bold",pad=10)
    for j in range(si+1):
        lbl,col=STEPS[j]
        r=mpatches.FancyBboxPatch((j-0.38,-0.35),0.76,0.9,boxstyle="round,pad=0.05",lw=2,edgecolor=col,facecolor=PALETTE["panel"])
        ax.add_patch(r)
        ax.text(j,0.1,lbl,ha="center",va="center",fontsize=9,color=PALETTE["text"],multialignment="center")
        if j<si: ax.annotate("",xy=(j+0.42,0.1),xytext=(j+0.58,0.1),arrowprops=dict(arrowstyle="->",color=col,lw=2))
    plt.tight_layout()
    for _ in range(6): gf.append(fig_to_rgb(fig))
    plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"01_pipeline_overview.gif"),gf,fps=4,loop=0)
print("GIF 01 done.")

# GIF 02 Frame diff normal
print("GIF 02...")
gf=[]; SI=list(range(0,N-1,6))
for i in SI:
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),facecolor=PALETTE["bg"])
    fig.suptitle("Frame Difference - Normal Video",color=PALETTE["text"],fontsize=12,fontweight="bold")
    p=cv2.cvtColor(nf[i],cv2.COLOR_BGR2GRAY).astype(float)
    c=cv2.cvtColor(nf[i+1],cv2.COLOR_BGR2GRAY).astype(float)
    axes[0].imshow(cv2.cvtColor(nf[i],cv2.COLOR_BGR2RGB)); axes[0].set_title(f"Frame {i}",color=PALETTE["subtext"]); axes[0].axis("off")
    axes[1].imshow(np.abs(c-p),cmap="hot",vmin=0,vmax=40); axes[1].set_title("Diff Heatmap",color=PALETTE["subtext"]); axes[1].axis("off")
    for ax in axes: ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout(); gf.append(fig_to_rgb(fig)); plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"02_frame_diff_normal.gif"),gf,fps=3,loop=0)
print("GIF 02 done.")

# GIF 03 Frame diff anomaly
print("GIF 03...")
gf=[]
for i in SI:
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),facecolor=PALETTE["bg"])
    fig.suptitle("Frame Difference - Anomaly Video",color=PALETTE["text"],fontsize=12,fontweight="bold")
    p=cv2.cvtColor(cf[i],cv2.COLOR_BGR2GRAY).astype(float)
    c=cv2.cvtColor(cf[i+1],cv2.COLOR_BGR2GRAY).astype(float)
    is_a=34<=i<=45
    col=PALETTE["accent3"] if is_a else PALETTE["accent1"]
    axes[0].imshow(cv2.cvtColor(cf[i],cv2.COLOR_BGR2RGB)); axes[0].set_title(f"Frame {i} [{'ANOMALY' if is_a else 'Normal'}]",color=col); axes[0].axis("off")
    axes[1].imshow(np.abs(c-p),cmap="hot",vmin=0,vmax=80); axes[1].set_title("Diff Heatmap",color=PALETTE["subtext"]); axes[1].axis("off")
    for ax in axes: ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout(); gf.append(fig_to_rgb(fig)); plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"03_frame_diff_anomaly.gif"),gf,fps=3,loop=0)
print("GIF 03 done.")

# GIF 04 Anomaly score animated
print("GIF 04...")
gf=[]; t=np.arange(N)
for end in list(range(4,N+1,4))+[N]:
    fig,ax=plt.subplots(figsize=(10,4),facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title("Anomaly Score Over Time",color=PALETTE["text"],fontsize=13,fontweight="bold")
    ax.set_xlabel("Frame",color=PALETTE["subtext"]); ax.set_ylabel("Frame-Diff Score",color=PALETTE["subtext"])
    ax.tick_params(colors=PALETTE["subtext"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["panel"])
    ax.set_xlim(0,N-1); ax.set_ylim(0,max(cs.max(),os_.max())*1.1)
    ax.axhline(THR,color=PALETTE["accent3"],lw=1.5,ls="--",label=f"Threshold={THR:.1f}")
    ax.plot(t[:end],ns[:end],color=PALETTE["accent2"],lw=2,label="Normal")
    ax.plot(t[:end],cs[:end],color=PALETTE["accent1"],lw=2,label="Crowd Anomaly")
    ax.plot(t[:end],os_[:end],color=PALETTE["accent3"],lw=2,label="Object Drop")
    ax.legend(facecolor=PALETTE["panel"],labelcolor=PALETTE["text"],fontsize=9)
    plt.tight_layout(); gf.append(fig_to_rgb(fig)); plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"04_anomaly_score_plot.gif"),gf,fps=6,loop=0)
print("GIF 04 done.")

# GIF 05 Detection overlay
print("GIF 05...")
gf=[]; atf=cf+of; ats=np.concatenate([cs,os_])
for i in range(0,2*N-1,5):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),facecolor=PALETTE["bg"])
    fig.suptitle("Anomaly Detection Overlay",color=PALETTE["text"],fontsize=13,fontweight="bold")
    raw=cv2.cvtColor(atf[i],cv2.COLOR_BGR2RGB).copy()
    sc=ats[i]; ov=raw.copy()
    if sc>THR:
        cv2.rectangle(ov,(5,5),(W-5,H-5),(220,38,38),4)
        cv2.putText(ov,f"ANOMALY {sc:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,38,38),2)
    else:
        cv2.putText(ov,f"NORMAL {sc:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(63,185,80),2)
    axes[0].imshow(raw); axes[0].set_title("Raw Frame",color=PALETTE["subtext"]); axes[0].axis("off")
    axes[1].imshow(ov);  axes[1].set_title("Detected Output",color=PALETTE["subtext"]); axes[1].axis("off")
    for ax in axes: ax.set_facecolor(PALETTE["bg"])
    plt.tight_layout(); gf.append(fig_to_rgb(fig)); plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"05_detection_overlay.gif"),gf,fps=5,loop=0)
print("GIF 05 done.")

# GIF 06 Confusion matrix
print("GIF 06...")
preds=(all_s>THR).astype(int)
TP=int(((preds==1)&(all_gt==1)).sum()); FP=int(((preds==1)&(all_gt==0)).sum())
FN=int(((preds==0)&(all_gt==1)).sum()); TN=int(((preds==0)&(all_gt==0)).sum())
pr=TP/(TP+FP+1e-9); rc=TP/(TP+FN+1e-9); f1=2*pr*rc/(pr+rc+1e-9)
print(f"  TP={TP} FP={FP} FN={FN} TN={TN} F1={f1:.2f}")
lbs=[[f"TN\n{TN}",f"FP\n{FP}"],[f"FN\n{FN}",f"TP\n{TP}"]]
cols=[[PALETTE["accent2"],PALETTE["accent3"]],[PALETTE["accent3"],PALETTE["accent2"]]]
gf=[]
for rev in range(5):
    fig,ax=plt.subplots(figsize=(5.5,4.5),facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title("Confusion Matrix",color=PALETTE["text"],fontsize=13,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: Normal","Pred: Anomaly"],color=PALETTE["text"])
    ax.set_yticklabels(["Actual: Normal","Actual: Anomaly"],color=PALETTE["text"])
    ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["panel"])
    for r in range(2):
        for c in range(2):
            if (r*2+c)<=rev:
                rct=mpatches.FancyBboxPatch((c-0.42,r-0.38),0.84,0.76,boxstyle="round,pad=0.04",lw=0,facecolor=cols[r][c]+"55")
                ax.add_patch(rct)
                ax.text(c,r,lbs[r][c],ha="center",va="center",fontsize=14,fontweight="bold",color=cols[r][c])
    ax.text(0.5,-0.52,f"F1={f1:.2f}  P={pr:.2f}  R={rc:.2f}",ha="center",color=PALETTE["subtext"],fontsize=10,transform=ax.transData)
    plt.tight_layout()
    for _ in range(5): gf.append(fig_to_rgb(fig))
    plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"06_confusion_matrix.gif"),gf,fps=2,loop=0)
print("GIF 06 done.")

# GIF 07 ROC
print("GIF 07...")
from sklearn.metrics import roc_curve, auc as auc_fn
fp2,tp2,_=roc_curve(all_gt,all_s)
ra=auc_fn(fp2,tp2)
gf=[]; Np=len(fp2)
steps=[*range(2,Np+1,max(1,Np//20)),Np]
for end in steps:
    fig,ax=plt.subplots(figsize=(6,5),facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.plot([0,1],[0,1],"--",color=PALETTE["subtext"],lw=1.2)
    ax.plot(fp2[:end],tp2[:end],color=PALETTE["accent1"],lw=2.5,label=f"AUC={ra:.3f}" if end==Np else "")
    ax.fill_between(fp2[:end],tp2[:end],alpha=0.15,color=PALETTE["accent1"])
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("False Positive Rate",color=PALETTE["subtext"])
    ax.set_ylabel("True Positive Rate",color=PALETTE["subtext"])
    ax.set_title("ROC Curve — Anomaly Detector",color=PALETTE["text"],fontsize=13,fontweight="bold")
    ax.tick_params(colors=PALETTE["subtext"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["panel"])
    if end==Np: ax.legend(facecolor=PALETTE["panel"],labelcolor=PALETTE["text"],fontsize=11)
    plt.tight_layout(); gf.append(fig_to_rgb(fig)); plt.close(fig)
imageio.mimsave(os.path.join(GIF_DIR,"07_roc_curve.gif"),gf,fps=8,loop=0)
print("GIF 07 done.")

print("ALL DONE!")
