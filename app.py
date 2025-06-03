import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de la página
st.set_page_config(page_title="Cálculo Simbólico", layout="wide")
st.title("App Maestría Matemática-UNACH")

# Símbolos matemáticos
x, y, z = sp.symbols("x y z")

# Menú lateral
opcion = st.sidebar.selectbox("Selecciona una operación:", [
    "Derivadas", "Integrales", "Matrices", "Sistemas lineales", "Sistemas cuadráticos"
])

if opcion == "Derivadas":
    st.header("🧮 Derivadas")
    with st.form(key="form_deriv"):
        func = st.text_input("Función a derivar:", "x**2 * sin(x)")
        var = st.selectbox("Variable:", ["x", "y", "z"])
        calcular = st.form_submit_button("Calcular")
        borrar = st.form_submit_button("Borrar")

    if calcular and func:
        try:
            expr = sp.sympify(func)
            derivada = sp.diff(expr, var)
            st.latex(f"\\frac{{d}}{{d{var}}}({sp.latex(expr)}) = {sp.latex(derivada)}")
            func_lambda = sp.lambdify(sp.Symbol(var), expr, modules=['numpy'])
            deriv_lambda = sp.lambdify(sp.Symbol(var), derivada, modules=['numpy'])
            xs = np.linspace(-10, 10, 400)
            ys = func_lambda(xs)
            dys = deriv_lambda(xs)
            fig, ax = plt.subplots()
            ax.plot(xs, ys, label="Función")
            ax.plot(xs, dys, label="Derivada")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

elif opcion == "Integrales":
    st.header("🔄 Integrales")
    with st.form(key="form_integ"):
        func = st.text_input("Función a integrar:", "x*sin(x)")
        tipo = st.radio("Tipo de integral:", ["Indefinida", "Definida"])
        a = st.number_input("Límite inferior", value=0.0, key="a") if tipo == "Definida" else None
        b = st.number_input("Límite superior", value=float(sp.pi.evalf()), key="b") if tipo == "Definida" else None
        calcular = st.form_submit_button("Calcular")
        borrar = st.form_submit_button("Borrar")

    if calcular and func:
        try:
            expr = sp.sympify(func)
            if tipo == "Indefinida":
                integral = sp.integrate(expr, x)
                st.latex(f"\\int {sp.latex(expr)} dx = {sp.latex(integral)} + C")
                xs = np.linspace(-10, 10, 400)
            else:
                integral = sp.integrate(expr, (x, a, b))
                st.latex(f"\\int_{{{a}}}^{{{b}}} {sp.latex(expr)} dx = {sp.latex(integral)}")
                xs = np.linspace(float(a), float(b), 400)
            func_lambda = sp.lambdify(x, expr, modules=['numpy'])
            ys = func_lambda(xs)
            fig, ax = plt.subplots()
            ax.plot(xs, ys, label="Función")
            if tipo == "Definida":
                ax.fill_between(xs, ys, alpha=0.3, color='orange', label="Área bajo la curva")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

elif opcion == "Matrices":
    st.header("📐 Operaciones con Matrices")
    
    # Configuración de la matriz
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Tamaño de la matriz (filas):", min_value=2, max_value=5, value=2)
    with col2:
        m = st.number_input("Tamaño de la matriz (columnas):", min_value=2, max_value=5, value=2)
    
    # Ingreso de la matriz A
    st.subheader("Matriz A")
    matriz_A = []
    for i in range(n):
        cols = st.columns(m)
        fila = []
        for j in range(m):
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0, key=f"A{i}{j}")
            fila.append(val)
        matriz_A.append(fila)
    A = sp.Matrix(matriz_A)
    
    # Ingreso de la matriz B (para operaciones binarias)
    st.subheader("Matriz B (para operaciones entre matrices)")
    matriz_B = []
    for i in range(n):
        cols = st.columns(m)
        fila = []
        for j in range(m):
            val = cols[j].number_input(f"B[{i+1},{j+1}]", value=1.0, key=f"B{i}{j}")
            fila.append(val)
        matriz_B.append(fila)
    B = sp.Matrix(matriz_B)
    
    # Mostrar matrices
    st.write("**Matriz A:**")
    st.latex(sp.latex(A))
    st.write("**Matriz B:**")
    st.latex(sp.latex(B))
    
    # Selección de operaciones
    operacion = st.selectbox("Seleccione una operación:", [
        "Propiedades básicas", 
        "Operaciones binarias", 
        "Descomposiciones", 
        "Visualización 3D"
    ])
    
    if operacion == "Propiedades básicas":
        st.subheader("Propiedades de la Matriz A")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Determinante:**")
            st.latex(sp.latex(A.det()))
            
            st.write("**Traza (suma diagonal):**")
            st.latex(sp.latex(A.trace()))
            
            st.write("**Transpuesta:**")
            st.latex(sp.latex(A.T))
            
        with col2:
            st.write("**Rango:**")
            st.latex(sp.latex(A.rank()))
            
            if A.det() != 0:
                st.write("**Inversa:**")
                st.latex(sp.latex(A.inv()))
            else:
                st.warning("La matriz no es invertible (determinante = 0)")
            
            st.write("**Forma escalonada reducida:**")
            st.latex(sp.latex(A.rref()[0]))
    
    elif operacion == "Operaciones binarias":
        st.subheader("Operaciones entre A y B")
        
        op_binaria = st.radio("Operación:", [
            "Suma (A+B)", 
            "Resta (A-B)", 
            "Multiplicación (A*B)", 
            "Producto Hadamard (A∘B)",
            "Producto Kronecker (A⊗B)"
        ])
        
        if op_binaria == "Suma (A+B)":
            resultado = A + B
        elif op_binaria == "Resta (A-B)":
            resultado = A - B
        elif op_binaria == "Multiplicación (A*B)":
            if A.cols == B.rows:
                resultado = A * B
            else:
                st.error("Las dimensiones no son compatibles para multiplicación")
                return
        elif op_binaria == "Producto Hadamard (A∘B)":
            if A.shape == B.shape:
                resultado = A.multiply_elementwise(B)
            else:
                st.error("Las matrices deben tener la misma dimensión")
                return
        elif op_binaria == "Producto Kronecker (A⊗B)":
            resultado = sp.kronecker_product(A, B)
        
        st.write(f"**Resultado de {op_binaria}:**")
        st.latex(sp.latex(resultado))
    
    elif operacion == "Descomposiciones":
        st.subheader("Descomposiciones Matriciales")
        
        descomp = st.selectbox("Seleccione descomposición:", [
            "LU", 
            "QR", 
            "Descomposición espectral"
        ])
        
        if descomp == "LU":
            try:
                L, U, _ = A.LUdecomposition()
                st.write("**Matriz triangular inferior (L):**")
                st.latex(sp.latex(L))
                st.write("**Matriz triangular superior (U):**")
                st.latex(sp.latex(U))
            except Exception as e:
                st.error(f"No se pudo realizar la descomposición LU: {e}")
        
        elif descomp == "QR":
            try:
                Q, R = A.QRdecomposition()
                st.write("**Matriz ortogonal (Q):**")
                st.latex(sp.latex(Q))
                st.write("**Matriz triangular superior (R):**")
                st.latex(sp.latex(R))
            except Exception as e:
                st.error(f"No se pudo realizar la descomposición QR: {e}")
        
        elif descomp == "Descomposición espectral":
            try:
                if A.is_symmetric():
                    eigenvects = A.eigenvects()
                    st.write("**Autovalores y autovectores:**")
                    for val, mult, vecs in eigenvects:
                        st.latex(f"\\lambda = {sp.latex(val)}")
                        st.write("Autovectores correspondientes:")
                        for v in vecs:
                            st.latex(sp.latex(v))
                else:
                    st.warning("La matriz no es simétrica (la descomposición espectral requiere una matriz simétrica)")
            except Exception as e:
                st.error(f"Error en la descomposición espectral: {e}")
    
    elif operacion == "Visualización 3D":
        st.subheader("Visualización de Matrices 3D")
        
        if n == 3 and m == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Convertir a numpy para visualización
            A_np = np.array(A).astype(float)
            
            # Vectores originales (ejes)
            orig = np.array([0, 0, 0])
            ejes = np.eye(3)
            
            # Vectores transformados
            transformados = A_np @ ejes
            
            # Dibujar ejes originales
            for i, color in enumerate(['r', 'g', 'b']):
                ax.quiver(*orig, *ejes[i], color=color, arrow_length_ratio=0.1, 
                         label=f'Eje {"XYZ"[i]} original', linewidth=2)
            
            # Dibujar vectores transformados
            for i, color in enumerate(['c', 'm', 'y']):
                ax.quiver(*orig, *transformados[i], color=color, arrow_length_ratio=0.1, 
                         label=f'Transformado {"XYZ"[i]}', linestyle='--')
            
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Transformación Lineal 3D')
            ax.legend()
            
            st.pyplot(fig)
            
            # Mostrar determinante como factor de escala volumétrico
            det = np.linalg.det(A_np)
            st.write(f"**Determinante (factor de escala volumétrico):** {det:.2f}")
            
            if abs(det) < 1e-5:
                st.warning("La transformación colapsa el espacio (determinante cercano a cero)")
        else:
            st.warning("La visualización 3D solo está disponible para matrices 3x3")

elif opcion == "Sistemas lineales":
    st.header("🧾 Sistema de Ecuaciones Lineales")
    with st.form(key="form_sist_lin"):
        eq1 = st.text_input("Ecuación 1", "x + y + z - 6")
        eq2 = st.text_input("Ecuación 2", "2*x - y + z - 3")
        eq3 = st.text_input("Ecuación 3", "-x + y + 2*z - 4")
        calcular = st.form_submit_button("Calcular")
        borrar = st.form_submit_button("Borrar")

    if calcular:
        try:
            sistema = [sp.sympify(eq1), sp.sympify(eq2), sp.sympify(eq3)]
            variables = list({v for eq in sistema for v in eq.free_symbols})
            soluciones = sp.solve(sistema, variables, dict=True)
            if soluciones:
                for i, sol in enumerate(soluciones):
                    st.write(f"Solución {i+1}: {sol}")
                if len(variables) == 3:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    xs, ys = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
                    for eq in sistema:
                        eq_z = sp.solve(eq, z)
                        if eq_z:
                            f = sp.lambdify((x, y), eq_z[0], 'numpy')
                            zs = f(xs, ys)
                            ax.plot_surface(xs, ys, zs, alpha=0.3)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    st.pyplot(fig)
            else:
                st.warning("No se encontraron soluciones únicas.")
        except Exception as e:
            st.error(f"Error: {e}")

elif opcion == "Sistemas cuadráticos":
    st.header("🔺 Sistema de Ecuaciones Cuadráticas")
    with st.form(key="form_sist_cuad"):
        eq1 = st.text_input("Ecuación 1:", "x**2 + y**2 - 4")
        eq2 = st.text_input("Ecuación 2:", "x*y - 1")
        calcular = st.form_submit_button("Calcular")
        borrar = st.form_submit_button("Borrar")

    if calcular:
        try:
            soluciones = sp.solve([sp.sympify(eq1), sp.sympify(eq2)], (x, y), dict=True)
            if soluciones:
                for i, sol in enumerate(soluciones):
                    st.write(f"Solución {i+1}: {sol}")
                x_vals = np.linspace(-5, 5, 400)
                y_vals = np.linspace(-5, 5, 400)
                X, Y = np.meshgrid(x_vals, y_vals)
                f1 = sp.lambdify((x, y), sp.sympify(eq1), 'numpy')
                f2 = sp.lambdify((x, y), sp.sympify(eq2), 'numpy')
                Z1 = f1(X, Y)
                Z2 = f2(X, Y)
                fig, ax = plt.subplots()
                CS1 = ax.contour(X, Y, Z1, levels=[0], colors='blue')
                CS2 = ax.contour(X, Y, Z2, levels=[0], colors='red')
                ax.clabel(CS1, inline=1, fontsize=10)
                ax.clabel(CS2, inline=1, fontsize=10)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                st.pyplot(fig)
            else:
                st.warning("No se encontraron soluciones.")
        except Exception as e:
            st.error(f"Error: {e}")

# Notas al pie
st.sidebar.markdown("---")
st.sidebar.info("""
**App de Cálculo Simbólico**  
Desarrollado para la Maestría en Matemáticas - UNACH  
Usando SymPy, NumPy y Streamlit
""")
