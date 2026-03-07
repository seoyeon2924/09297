"""
LangGraph 워크플로우 시각화 테스트.
Mermaid 다이어그램을 출력하고, PNG 저장을 시도한다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains.review_chain import graph


def main() -> None:
    g = graph.get_graph()

    print("=== 노드 ===")
    for node in g.nodes:
        print(f"  - {node}")

    print("\n=== 엣지 ===")
    for edge in g.edges:
        print(f"  {edge}")

    print("\n=== Mermaid 다이어그램 ===")
    mermaid_code = g.draw_mermaid()
    print(mermaid_code)
    print("\n위 코드를 https://mermaid.live 에 붙여넣으면 그래프를 볼 수 있습니다.")

    try:
        png_bytes = g.draw_mermaid_png()
        out_path = ROOT / "graph_structure.png"
        out_path.write_bytes(png_bytes)
        print(f"\nPNG 저장 완료: {out_path}")
    except Exception as e:
        print(f"\nPNG 저장 실패 (무시 가능): {e}")


if __name__ == "__main__":
    main()
