"""Regenerates all four figures from the verified numbers (no rounding beyond what is printed)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
BLUE,RED,GREEN,GREY="#2b6ea8","#b23a48","#2e8b57","#777777"

def forest(ax, rows, colors, xlabel):
    for i,(lab,m,lo,hi) in enumerate(rows):
        y=len(rows)-1-i
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt="o",color=colors[i],capsize=2.5,ms=4,lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0,color=GREY,ls="--",lw=0.8); ax.set_xlabel(xlabel)

# ---- Fig 1: estimator constructions on identical 225 starts + paired contrast
rows=[("Rollout: pooled episodes",-0.0895,-0.2228,0.0265),
      ("Rollout: seed $t$ (n=10)",-0.0895,-0.3192,0.1402),
      ("Rollout: hierarchical",-0.0895,-0.3621,0.0984),
      ("FQE: seed $t$ (n=10)",0.3478,0.0882,0.6075),
      ("FQE: state bootstrap",0.3478,0.2745,0.4389),
      ("FQE $-$ rollout (paired seeds)",0.4373,0.1131,0.7615)]
fig,ax=plt.subplots(figsize=(3.5,2.5))
forest(ax,rows,[BLUE]*3+[RED]*2+["k"],"CQL $-$ BC (return units)")
fig.tight_layout(); fig.savefig("figs/fig1_estimators.pdf"); plt.close(fig)

# ---- Fig 2: 9 pairings x 3 start sets
P=[("base","base",(0.361,0.250,0.473),(-2.315,-2.965,-1.614),(-2.255,-2.789,-1.728)),
   ("base","epsilon",(0.446,0.329,0.573),(-2.707,-3.226,-2.174),(-2.513,-3.003,-2.049)),
   ("base","no_oracle",(0.407,0.302,0.519),(-2.510,-3.052,-1.950),(-2.421,-2.986,-1.856)),
   ("epsilon","base",(-0.098,-0.187,-0.016),(0.302,-0.066,0.694),(0.302,-0.066,0.694)),
   ("epsilon","epsilon",(-0.013,-0.101,0.069),(-0.089,-0.362,0.098),(-0.089,-0.362,0.098)),
   ("epsilon","no_oracle",(-0.052,-0.151,0.054),(0.108,-0.132,0.356),(0.108,-0.132,0.356)),
   ("no_oracle","base",(0.263,0.062,0.480),(-0.277,-0.754,0.145),(-0.325,-0.861,0.158)),
   ("no_oracle","epsilon",(0.348,0.159,0.558),(-0.669,-1.182,-0.209),(-0.645,-1.186,-0.127)),
   ("no_oracle","no_oracle",(0.309,0.166,0.480),(-0.472,-0.914,-0.039),(-0.461,-0.957,-0.020))]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
for k,(ax,t) in enumerate(zip(axs,["RF-grid (500)","Common epsilon test (225)","Own-family test"])):
    for i,p in enumerate(P):
        m,lo,hi=p[2+k]; y=8-i
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt="o",ms=3.5,capsize=2,lw=1.1,
                    color=BLUE if p[0]=="base" else (RED if p[0]=="epsilon" else GREEN))
    ax.axvline(0,color=GREY,ls="--",lw=.8); ax.set_title(t,fontsize=8); ax.set_xlabel("CQL $-$ BC")
axs[0].set_yticks(range(9)); axs[0].set_yticklabels([f"{p[0]} / {p[1]}" for p in P][::-1],fontsize=7)
axs[0].set_ylabel("BC family / CQL family")
from matplotlib.lines import Line2D
axs[2].legend(handles=[Line2D([],[],marker="o",ls="",color=c,label=l) for c,l in
              [(BLUE,"base BC"),(RED,"epsilon BC"),(GREEN,"no-oracle BC")]],fontsize=6.5,loc="lower right")
fig.tight_layout(); fig.savefig("figs/fig2_pairings.pdf"); plt.close(fig)

# ---- Fig 3: BC sweep
S={"base":[(.369,.250,.497),(.203,.069,.355),(.086,-.055,.238),(.076,-.073,.227),(.267,.118,.397),(.257,.116,.399)],
   "epsilon":[(-.057,-.128,.004),(-.066,-.134,-.005),(-.049,-.076,-.020),(-.063,-.109,-.017),(-.039,-.079,-.0002),(-.044,-.094,-.0002)],
   "no_oracle":[(.221,.062,.409),(.234,-.004,.544),(.262,.032,.507),(.165,-.001,.336),(.186,.050,.314),(.066,-.051,.182)]}
cells=[("$10^{-3}$","256"),("$10^{-3}$","128"),("$3{\\times}10^{-4}$","256"),("$3{\\times}10^{-4}$","128"),("$10^{-4}$","256"),("$10^{-4}$","128")]
fig,ax=plt.subplots(figsize=(3.5,2.6)); off={"base":-.2,"epsilon":0,"no_oracle":.2}
for fam,c,mk in [("base",BLUE,"o"),("epsilon",RED,"s"),("no_oracle",GREEN,"^")]:
    for i,(m,lo,hi) in enumerate(S[fam]):
        ax.errorbar(i+off[fam],m,yerr=[[m-lo],[hi-m]],fmt=mk,color=c,ms=3.5,capsize=2,lw=1,label=fam.replace("_","-") if i==0 else None)
ax.axhline(0,color=GREY,ls="--",lw=.8); ax.set_xticks(range(6))
ax.set_xticklabels([f"{a}\n{b}" for a,b in cells],fontsize=6.5); ax.set_ylabel("CQL $-$ BC (RF-grid)")
ax.set_xlabel("BC learning rate / hidden width"); ax.legend(fontsize=6.5,frameon=False,loc="upper right")
fig.tight_layout(); fig.savefig("figs/fig3_bc_sweep.pdf"); plt.close(fig)

# ---- Fig 4: band gaps, w3 = 0 vs 0.5
bands=["in-dist","mild","moderate","severe"]
z=[(1.0182,.0016,1.9506),(3.9334,2.2139,5.5946),(6.5652,3.4935,9.7477),(6.3407,2.5408,10.3509)]
f=[(1.0687,-.0074,2.0566),(4.1222,2.3105,5.8742),(6.8796,3.6911,10.1810),(6.6664,2.7780,10.7590)]
fig,ax=plt.subplots(figsize=(3.5,2.5))
for d,c,o,l in [(z,BLUE,-.06,"$w_3=0$ (removed)"),(f,RED,.06,"$w_3=0.5$ (full)")]:
    for i,(m,lo,hi) in enumerate(d):
        ax.errorbar(i+o,m,yerr=[[m-lo],[hi-m]],fmt="o",color=c,ms=3.5,capsize=2,lw=1,label=l if i==0 else None)
ax.axhline(0,color=GREY,ls="--",lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels(bands)
ax.set_ylabel("CQL $-$ BC (return)"); ax.legend(fontsize=6.5,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig4_bands.pdf"); plt.close(fig)
print("ok")