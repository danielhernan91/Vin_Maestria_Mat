import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Cálculo Simbólico", layout="wide")
st.title("App Maestria Matemática-UNACH")

x, y, z = sp.symbols("x y z")

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
    n = st.number_input("Tamaño de la matriz cuadrada:", min_value=2, max_value=5, value=2)
    matriz = []
    for i in range(n):
        cols = st.columns(n)
        fila = []
        for j in range(n):
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=0.0, key=f"A{i}{j}")
            fila.append(val)
        matriz.append(fila)
    A = sp.Matrix(matriz)
    st.write("**Matriz ingresada:**")
    st.latex(sp.latex(A))
    st.write("**Determinante:**", A.det())
    if A.det() != 0:
        st.write("**Inversa:**")
        st.latex(sp.latex(A.inv()))
    st.write("**Rango:**", A.rank())

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
