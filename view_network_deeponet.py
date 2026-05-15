import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_minimal_deeponet(save_path='deeponet_minimal.png'):
    """Предельно минималистичная схема DeepONet"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Только черный и оттенки серого
    c1 = '#333333'  # темно-серый
    c2 = '#666666'  # средне-серый
    c3 = '#999999'  # светло-серый
    
    # Заголовок
    ax.text(6, 5.7, 'DeepONet + PINN', ha='center', fontsize=13, fontweight='bold', color=c1)
    
    # ═══════════ BRANCH ═══════════
    
    ax.text(2.5, 5.2, 'BRANCH', ha='center', fontsize=10, fontweight='bold', color=c1)
    
    boxes_left = [
        (1.0, 4.5, 3.0, 0.4, 'Вход: 25 сенсоров'),
        (1.0, 3.8, 3.0, 0.4, '128 → 128 → 128'),
        (1.0, 3.1, 3.0, 0.4, 'b (128)'),
    ]
    
    for x, y, w, h, text in boxes_left:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=c1, facecolor='white')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, color=c1)
    
    # Стрелки branch
    for y1, y2 in [(4.5, 4.2), (3.8, 3.5)]:
        ax.annotate('', xy=(2.5, y2+0.05), xytext=(2.5, y1-0.05),
                    arrowprops=dict(arrowstyle='->', lw=1, color=c1))
    
    # ═══════════ TRUNK ═══════════
    
    ax.text(9.5, 5.2, 'TRUNK', ha='center', fontsize=10, fontweight='bold', color=c1)
    
    boxes_right = [
        (8.0, 4.5, 3.0, 0.4, 'Вход: (x,y) + Fourier'),
        (8.0, 3.8, 3.0, 0.4, '128 → 128 → 128'),
        (8.0, 3.1, 3.0, 0.4, 't (128)'),
    ]
    
    for x, y, w, h, text in boxes_right:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=c1, facecolor='white')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, color=c1)
    
    # Стрелки trunk
    for y1, y2 in [(4.5, 4.2), (3.8, 3.5)]:
        ax.annotate('', xy=(9.5, y2+0.05), xytext=(9.5, y1-0.05),
                    arrowprops=dict(arrowstyle='->', lw=1, color=c1))
    
    # ═══════════ СТРЕЛКИ К ЦЕНТРУ ═══════════
    
    ax.annotate('', xy=(6, 2.3), xytext=(4, 3.3),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c2, connectionstyle="arc3,rad=0.3"))
    ax.annotate('', xy=(6, 2.3), xytext=(8, 3.3),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c2, connectionstyle="arc3,rad=-0.3"))
    
    # ═══════════ ПРОИЗВЕДЕНИЕ ═══════════
    
    rect = patches.Rectangle((4, 1.7), 4, 1.2, linewidth=1.5, edgecolor=c1, facecolor='white')
    ax.add_patch(rect)
    ax.text(6, 2.4, 'b · t', ha='center', fontsize=12, fontweight='bold', color=c1)
    ax.text(6, 2.1, 'Поэлементное произведение', ha='center', fontsize=8, color=c2)
    ax.text(6, 1.85, '→ 128 → 64 → 1 →', ha='center', fontsize=8, color=c2)
    
    # ═══════════ ВЫХОДЫ ═══════════
    
    outputs = [
        (2.5, 'u', 'Ux'),
        (6.0, 'v', 'Uy'),
        (9.5, 'p', 'P'),
    ]
    
    for x, title, sub in outputs:
        ax.annotate('', xy=(x, 1.0), xytext=(6, 1.7),
                    arrowprops=dict(arrowstyle='->', lw=1, color=c3))
        
        rect = patches.Rectangle((x-1, 0.6), 2, 0.8, linewidth=1, edgecolor=c1, facecolor='white')
        ax.add_patch(rect)
        ax.text(x, 1.1, title, ha='center', fontsize=12, fontweight='bold', color=c1)
        ax.text(x, 0.8, sub, ha='center', fontsize=8, color=c2)
    
    # ═══════════ ФИЗИКА ═══════════
    
    rect = patches.Rectangle((0.2, 0.6), 1.8, 1.5, linewidth=1, edgecolor=c2, facecolor='white', linestyle='--')
    ax.add_patch(rect)
    ax.text(1.1, 1.8, 'Loss', ha='center', fontsize=9, fontweight='bold', color=c1)
    ax.text(1.1, 1.5, 'Mx + My', ha='center', fontsize=7, color=c2)
    ax.text(1.1, 1.3, '+ ∇·u = 0', ha='center', fontsize=7, color=c2)
    
    ax.annotate('', xy=(1.1, 2.1), xytext=(1.1, 2.6),
                arrowprops=dict(arrowstyle='->', lw=0.8, color=c3, linestyle='--'))
    
    # ═══════════ ПОДПИСЬ ═══════════
    
    ax.text(6, 0.15, '3 сети × 3 слоя × 128 нейронов | GELU | LayerNorm | Dropout', 
            ha='center', fontsize=7, color=c3)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()


draw_minimal_deeponet('deeponet_minimal.png')