import random
import sys
import numpy as np
import pygame

# ---------------- Configuración ----------------
FILAS, COLUMNAS, TAM = 20, 10, 30
PANEL = 200
ANCHO, ALTO = COLUMNAS * TAM + PANEL, FILAS * TAM

# Cada pieza es una matriz; sus celdas ocupadas valen el id del color (1..7)
FORMAS = {
    1: np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),  # I
    2: np.array([[2, 2], [2, 2]]),                                          # O
    3: np.array([[0, 3, 0], [3, 3, 3], [0, 0, 0]]),                         # T
    4: np.array([[0, 4, 4], [4, 4, 0], [0, 0, 0]]),                         # S
    5: np.array([[5, 5, 0], [0, 5, 5], [0, 0, 0]]),                         # Z
    6: np.array([[6, 0, 0], [6, 6, 6], [0, 0, 0]]),                         # J
    7: np.array([[0, 0, 7], [7, 7, 7], [0, 0, 0]]),                         # L
}

COLORES = {
    0: (25, 25, 35),
    1: (0, 220, 230), 2: (240, 220, 0), 3: (170, 0, 220), 4: (0, 210, 60),
    5: (230, 30, 30), 6: (40, 60, 240), 7: (250, 150, 0),
}

PUNTOS_LINEAS = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}


# ---------------- Operaciones de álgebra lineal ----------------
def rotar_horario(m):
    """Rot(A) = A^T · J  (transponer e invertir el orden de las columnas)."""
    return m.T[:, ::-1]


def rotar_antihorario(m):
    """Transponer e invertir el orden de las filas."""
    return m.T[::-1, :]


class Juego:
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.tablero = np.zeros((FILAS, COLUMNAS), dtype=int)
        self.bolsa = []
        self.puntos = 0
        self.lineas = 0
        self.nivel = 1
        self.terminado = False
        self.siguiente = self.sacar_de_bolsa()
        self.nueva_pieza()

    def sacar_de_bolsa(self):
        if not self.bolsa:
            self.bolsa = list(FORMAS.keys())
            random.shuffle(self.bolsa)
        return self.bolsa.pop()

    def nueva_pieza(self):
        self.forma = FORMAS[self.siguiente].copy()
        self.siguiente = self.sacar_de_bolsa()
        self.fila = 0
        self.col = COLUMNAS // 2 - self.forma.shape[1] // 2
        if self.colision(self.forma, self.fila, self.col):
            self.terminado = True

    def colision(self, forma, fila, col):
        """True si alguna celda ocupada de la pieza sale del tablero
        o coincide con una celda ocupada (equivale a Hadamard != 0)."""
        for r, c in np.argwhere(forma > 0):
            rr, cc = fila + r, col + c
            if cc < 0 or cc >= COLUMNAS or rr >= FILAS:
                return True
            if rr >= 0 and self.tablero[rr, cc] != 0:
                return True
        return False

    def mover(self, dfila, dcol):
        if not self.colision(self.forma, self.fila + dfila, self.col + dcol):
            self.fila += dfila
            self.col += dcol
            return True
        return False

    def rotar(self, sentido):
        nueva = rotar_horario(self.forma) if sentido > 0 else rotar_antihorario(self.forma)
        for desplazamiento in (0, -1, 1, -2, 2):  # "wall kick" sencillo
            if not self.colision(nueva, self.fila, self.col + desplazamiento):
                self.forma = np.ascontiguousarray(nueva)
                self.col += desplazamiento
                return True
        return False

    def fijar(self):
        """Suma la pieza al tablero (B = S + T)."""
        for r, c in np.argwhere(self.forma > 0):
            self.tablero[self.fila + r, self.col + c] = self.forma[r, c]

    def limpiar_lineas(self):
        """Una fila está llena si todas sus entradas son distintas de 0
        (equivale a que la suma de la fila de la matriz 0/1 sea = COLUMNAS)."""
        llenas = np.all(self.tablero != 0, axis=1)
        n = int(llenas.sum())
        if n:
            restantes = self.tablero[~llenas]
            vacias = np.zeros((n, COLUMNAS), dtype=int)
            self.tablero = np.vstack((vacias, restantes))
            self.lineas += n
            self.puntos += PUNTOS_LINEAS[n] * self.nivel
            self.nivel = 1 + self.lineas // 10

    def bloquear(self):
        self.fijar()
        self.limpiar_lineas()
        self.nueva_pieza()

    def bajar(self):
        if not self.mover(1, 0):
            self.bloquear()

    def caida_rapida(self):
        while self.mover(1, 0):
            self.puntos += 2
        self.bloquear()

    def fila_fantasma(self):
        f = self.fila
        while not self.colision(self.forma, f + 1, self.col):
            f += 1
        return f

    def intervalo(self):
        return max(80, 800 - (self.nivel - 1) * 70)


# ---------------- Dibujo ----------------
def dibujar_celda(pantalla, x, y, color, relleno=True):
    rect = pygame.Rect(x, y, TAM - 1, TAM - 1)
    if relleno:
        pygame.draw.rect(pantalla, color, rect)
    else:
        pygame.draw.rect(pantalla, color, rect, 2)


def dibujar(pantalla, fuente, grande, juego, pausa):
    pantalla.fill((12, 12, 18))

    # Tablero
    for r in range(FILAS):
        for c in range(COLUMNAS):
            dibujar_celda(pantalla, c * TAM, r * TAM, COLORES[juego.tablero[r, c]])

    if not juego.terminado:
        # Pieza fantasma
        ff = juego.fila_fantasma()
        for r, c in np.argwhere(juego.forma > 0):
            color = COLORES[juego.forma[r, c]]
            dibujar_celda(pantalla, (juego.col + c) * TAM, (ff + r) * TAM, color, False)
        # Pieza actual
        for r, c in np.argwhere(juego.forma > 0):
            color = COLORES[juego.forma[r, c]]
            dibujar_celda(pantalla, (juego.col + c) * TAM, (juego.fila + r) * TAM, color)

    # Panel lateral
    x0 = COLUMNAS * TAM + 20
    textos = [f"Puntos: {juego.puntos}", f"Lineas: {juego.lineas}", f"Nivel: {juego.nivel}"]
    for i, t in enumerate(textos):
        pantalla.blit(fuente.render(t, True, (230, 230, 230)), (x0, 20 + i * 30))

    pantalla.blit(fuente.render("Siguiente:", True, (230, 230, 230)), (x0, 140))
    sig = FORMAS[juego.siguiente]
    for r, c in np.argwhere(sig > 0):
        dibujar_celda(pantalla, x0 + c * TAM, 175 + r * TAM, COLORES[sig[r, c]])

    ayuda = ["<- -> mover", "abajo: bajar", "Arriba/X: rotar", "Z: rotar inverso",
             "Espacio: caer", "P: pausa", "R: reiniciar"]
    for i, t in enumerate(ayuda):
        pantalla.blit(fuente.render(t, True, (150, 150, 160)), (x0, 330 + i * 24))

    if juego.terminado or pausa:
        msg = "FIN DEL JUEGO" if juego.terminado else "PAUSA"
        sup = grande.render(msg, True, (255, 255, 255))
        rect = sup.get_rect(center=(COLUMNAS * TAM // 2, ALTO // 2))
        pygame.draw.rect(pantalla, (0, 0, 0), rect.inflate(30, 20))
        pantalla.blit(sup, rect)

    pygame.display.flip()


# ---------------- Programa principal ----------------
def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Tetris con matrices - Álgebra Lineal")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont("consolas", 20)
    grande = pygame.font.SysFont("consolas", 28, bold=True)

    juego = Juego()
    pausa = False
    t_caida = t_lateral = t_abajo = pygame.time.get_ticks()

    while True:
        ahora = pygame.time.get_ticks()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    juego.reiniciar()
                    pausa = False
                elif e.key == pygame.K_p:
                    pausa = not pausa
                elif not pausa and not juego.terminado:
                    if e.key in (pygame.K_UP, pygame.K_x):
                        juego.rotar(+1)
                    elif e.key == pygame.K_z:
                        juego.rotar(-1)
                    elif e.key == pygame.K_SPACE:
                        juego.caida_rapida()
                        t_caida = ahora

        if not pausa and not juego.terminado:
            teclas = pygame.key.get_pressed()
            if ahora - t_lateral > 110:
                if teclas[pygame.K_LEFT]:
                    juego.mover(0, -1)
                    t_lateral = ahora
                elif teclas[pygame.K_RIGHT]:
                    juego.mover(0, 1)
                    t_lateral = ahora
            if teclas[pygame.K_DOWN] and ahora - t_abajo > 50:
                if juego.mover(1, 0):
                    juego.puntos += 1
                t_abajo = ahora
            if ahora - t_caida > juego.intervalo():
                juego.bajar()
                t_caida = ahora

        dibujar(pantalla, fuente, grande, juego, pausa)
        reloj.tick(60)

if __name__ == "__main__":
    main()

# Rotación: Rot(A) = Aᵀ · J
def rotar_horario(m):
    return m.T[:, ::-1]

# Colisión: ¿alguna celda de la pieza coincide con una ocupada del tablero?
def colision(self, forma, fila, col):
    for r, c in np.argwhere(forma > 0):
        rr, cc = fila + r, col + c
        if cc < 0 or cc >= COLUMNAS or rr >= FILAS:
            return True
        if rr >= 0 and self.tablero[rr, cc] != 0:
            return True
    return False

# Fijar la pieza: B = S + T
self.tablero[self.fila + r, self.col + c] = self.forma[r, c]

# Líneas llenas y desplazamiento de filas superiores
llenas = np.all(self.tablero != 0, axis=1)
restantes = self.tablero[~llenas]
vacias = np.zeros((n, COLUMNAS), dtype=int)
self.tablero = np.vstack((vacias, restantes))
