"""
SISTEMA INTELIGENTE DE ETIQUETADO
Ofrece 3 estrategias de etiquetado con análisis de trade-offs
"""
from pathlib import Path
import json
import random
from datetime import datetime

class IntelligentLabeling:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.images_dir = self.project_root / "data" / "processed" / "images" / "train"
        self.classes = ["persona", "computadora", "teclado", "mouse", "telefono", "libro_documento", "taza_cafe"]
    
    def analyze_labeling_needs(self):
        """Analiza las necesidades de etiquetado basado en los frames extraídos"""
        images = list(self.images_dir.glob("*.jpg"))
        
        if not images:
            print("❌ No hay imágenes para etiquetar")
            return None
        
        analysis = {
            'total_images': len(images),
            'estimated_labeling_time': {
                'complete': len(images) * 0.5,  # 30 segundos por imagen
                'strategic': min(150, len(images)) * 0.5,
                'batch': len(images) * 0.3  # Más rápido con atajos
            },
            'recommendation': self._get_recommendation(len(images))
        }
        
        return analysis
    
    def _get_recommendation(self, total_images):
        """Genera recomendación basada en el número de imágenes"""
        if total_images <= 100:
            return {
                'strategy': 'complete',
                'reason': 'Pocas imágenes, mejor calidad máxima',
                'time_estimate': f"{total_images * 0.5 / 60:.1f} horas"
            }
        elif total_images <= 300:
            return {
                'strategy': 'strategic', 
                'reason': 'Balance ideal entre tiempo y calidad',
                'time_estimate': f"{min(150, total_images) * 0.5 / 60:.1f} horas"
            }
        else:
            return {
                'strategy': 'batch',
                'reason': 'Muchas imágenes, enfoque eficiente',
                'time_estimate': f"{total_images * 0.3 / 60:.1f} horas"
            }
    
    def present_labeling_options(self, analysis):
        """Presenta las 3 opciones de etiquetado con análisis detallado"""
        print("\n" + "="*70)
        print("🎯 SISTEMA INTELIGENTE DE ETIQUETADO")
        print("="*70)
        print(f"📊 Análisis: Tienes {analysis['total_images']} imágenes para etiquetar")
        print(f"💡 Recomendación: {analysis['recommendation']['strategy'].upper()} - {analysis['recommendation']['reason']}")
        print(f"⏱️  Tiempo estimado: {analysis['recommendation']['time_estimate']}")
        print("\n" + "─"*70)
        
        options = {
            '1': {
                'name': 'ETIQUETADO COMPLETO',
                'description': 'Máxima calidad - Etiquetar TODAS las imágenes',
                'frames': analysis['total_images'],
                'time': f"{analysis['estimated_labeling_time']['complete'] / 60:.1f} horas",
                'pros': [
                    '✅ Máxima precisión del modelo',
                    '✅ Dataset balanceado y completo', 
                    '✅ Mejor generalización',
                    '✅ Evita overfitting'
                ],
                'cons': [
                    '⏰ Más tiempo requerido',
                    '🔄 Puede ser repetitivo',
                    '💤 Mayor fatiga'
                ],
                'best_for': 'Modelos de producción, investigación, máxima calidad'
            },
            '2': {
                'name': 'ETIQUETADO ESTRATÉGICO', 
                'description': 'Balance ideal - Selección inteligente de imágenes',
                'frames': min(150, analysis['total_images']),
                'time': f"{analysis['estimated_labeling_time']['strategic'] / 60:.1f} horas",
                'pros': [
                    '⚡ 70% más rápido que completo',
                    '🎯 Enfocado en imágenes variadas',
                    '📈 Buen balance calidad/tiempo',
                    '🔍 Evita redundancia'
                ],
                'cons': [
                    '📉 Leve reducción en precisión',
                    '🎲 Requiere selección cuidadosa',
                    '⚠️ Posible sesgo en selección'
                ],
                'best_for': 'Prototipos rápidos, proyectos con tiempo limitado'
            },
            '3': {
                'name': 'ETIQUETADO POR LOTES',
                'description': 'Máxima eficiencia - Flujo rápido con atajos',
                'frames': analysis['total_images'], 
                'time': f"{analysis['estimated_labeling_time']['batch'] / 60:.1f} horas",
                'pros': [
                    '🚀 40% más rápido que estratégico',
                    '⌨️ Uso intensivo de atajos de teclado',
                    '📁 Procesamiento por lotes',
                    '🔄 Flujo continuo sin pausas'
                ],
                'cons': [
                    '📉 Mayor riesgo de errores',
                    '👁️ Menor atención a detalles',
                    '🔧 Curva de aprendizaje de atajos'
                ],
                'best_for': 'Experienced users, large datasets, time-critical projects'
            }
        }
        
        # Mostrar opciones
        for key, option in options.items():
            print(f"\n{key}. {option['name']}")
            print(f"   📝 {option['description']}")
            print(f"   📊 Frames: {option['frames']} | ⏱️  Tiempo: {option['time']}")
            print(f"   ✅ Ventajas: {', '.join(option['pros'][:2])}")
            print(f"   ⚠️  Consideraciones: {', '.join(option['cons'][:2])}")
            print(f"   🎯 Ideal para: {option['best_for']}")
        
        return options
    
    def execute_labeling_strategy(self, strategy, analysis):
        """Ejecuta la estrategia de etiquetado seleccionada"""
        print(f"\n🎯 EJECUTANDO ESTRATEGIA: {strategy.upper()}")
        print("─" * 50)
        
        if strategy == 'complete':
            self._setup_complete_labeling(analysis)
        elif strategy == 'strategic':
            self._setup_strategic_labeling(analysis) 
        elif strategy == 'batch':
            self._setup_batch_labeling(analysis)
        else:
            print("❌ Estrategia no válida")
            return
        
        self._launch_labeling_tool()
    
    def _setup_complete_labeling(self, analysis):
        """Prepara etiquetado completo"""
        print("📋 CONFIGURACIÓN - ETIQUETADO COMPLETO")
        print(f"   🎯 Objetivo: Etiquetar {analysis['total_images']} imágenes")
        print(f"   ⏱️  Tiempo estimado: {analysis['estimated_labeling_time']['complete'] / 60:.1f} horas")
        print("   💡 Consejos:")
        print("      • Toma descansos cada 30 minutos")
        print("      • Usa atajos: 'W' (bbox), 'D' (siguiente), 'A' (anterior)")
        print("      • Mantén consistencia en los bounding boxes")
        print("      • Guarda frecuentemente (Ctrl + S)")
    
    def _setup_strategic_labeling(self, analysis):
        """Prepara etiquetado estratégico"""
        target_images = min(150, analysis['total_images'])
        selected_images = self._select_strategic_images(target_images)
        
        print("📋 CONFIGURACIÓN - ETIQUETADO ESTRATÉGICO") 
        print(f"   🎯 Objetivo: Etiquetar {target_images} de {analysis['total_images']} imágenes")
        print(f"   ⏱️  Tiempo estimado: {analysis['estimated_labeling_time']['strategic'] / 60:.1f} horas")
        print(f"   📈 Eficiencia: {((1 - target_images/analysis['total_images']) * 100):.1f}% de ahorro")
        print("   🔍 Criterios de selección:")
        print("      • Variedad de ángulos y composiciones")
        print("      • Diferentes condiciones de iluminación")
        print("      • Objetos claramente visibles")
        print("      • Evitar frames casi idénticos")
        
        # Guardar lista de imágenes seleccionadas
        self._save_selected_images(selected_images, 'strategic')
    
    def _setup_batch_labeling(self, analysis):
        """Prepara etiquetado por lotes"""
        print("📋 CONFIGURACIÓN - ETIQUETADO POR LOTES")
        print(f"   🎯 Objetivo: Etiquetar {analysis['total_images']} imágenes eficientemente")
        print(f"   ⏱️  Tiempo estimado: {analysis['estimated_labeling_time']['batch'] / 60:.1f} horas")
        print("   ⌨️  ATAJOS CLAVE:")
        print("      • W → Crear bounding box")
        print("      • D → Siguiente imagen") 
        print("      • A → Imagen anterior")
        print("      • Ctrl + S → Guardar y continuar")
        print("      • Espacio → Guardar sin avanzar")
        print("      • Ctrl + D → Duplicar bounding box anterior")
        print("   🎯 Estrategia:")
        print("      • Enfocarse en velocidad sobre perfección")
        print("      • Revisar y corregir después del primer paso")
        print("      • Usar el mismo tamaño de bbox para objetos similares")
    
    def _select_strategic_images(self, target_count):
        """Selecciona imágenes estratégicas para etiquetado"""
        all_images = list(self.images_dir.glob("*.jpg"))
        
        if len(all_images) <= target_count:
            return all_images
        
        # Estrategia: muestreo distribuido + aleatoriedad
        step = len(all_images) // target_count
        selected = all_images[::step][:target_count//2]
        
        # Agregar aleatoriedad para variedad
        remaining = [img for img in all_images if img not in selected]
        selected.extend(random.sample(remaining, target_count - len(selected)))
        
        return selected
    
    def _save_selected_images(self, images, strategy):
        """Guarda la lista de imágenes seleccionadas"""
        selection_dir = self.project_root / "data" / "analysis" / "labeling"
        selection_dir.mkdir(parents=True, exist_ok=True)
        
        selection_file = selection_dir / f"labeling_selection_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        selection_data = {
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(images),
            'images': [str(img.name) for img in images],
            'classes': self.classes
        }
        
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump(selection_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Lista guardada: {selection_file}")
    
    def _launch_labeling_tool(self):
        """Inicia la herramienta de etiquetado"""
        print("\n🚀 INICIANDO HERRAMIENTA DE ETIQUETADO...")
        print("─" * 50)
        print("📋 INSTRUCCIONES FINALES:")
        print("   1. Abre LabelImg con: python -m labelImg")
        print("   2. Ve a: data/processed/images/train")
        print("   3. Configura formato YOLO")
        print("   4. ¡Comienza a etiquetar!")
        print("\n💡 ¿Necesitas ayuda? Ejecuta: python -m labelImg --help")
        
        # Preguntar si abrir LabelImg ahora
        launch_now = input("\n¿Abrir LabelImg ahora? (s/n): ").strip().lower()
        if launch_now == 's':
            try:
                import subprocess
                subprocess.run(["python", "-m", "labelImg", str(self.images_dir)])
            except Exception as e:
                print(f"❌ Error abriendo LabelImg: {e}")
                print("💡 Abre manualmente con: python -m labelImg")

def main():
    labeling_system = IntelligentLabeling()
    
    # Analizar necesidades
    analysis = labeling_system.analyze_labeling_needs()
    if not analysis:
        return
    
    # Presentar opciones
    options = labeling_system.present_labeling_options(analysis)
    
    # Selección de estrategia
    choice = input("\n🎯 Selecciona estrategia de etiquetado (1-3): ").strip()
    
    if choice in options:
        strategy_map = {'1': 'complete', '2': 'strategic', '3': 'batch'}
        labeling_system.execute_labeling_strategy(strategy_map[choice], analysis)
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()