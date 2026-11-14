# Part 1 — What is a chip package, and why do chips need wires?

*Series: Wire-Bond Defects, from First Principles to AI — Part 1 of 7*

---

## The problem nobody outside the fab thinks about

A modern silicon chip is a marvel of miniaturisation — billions of transistors on a sliver of
silicon smaller than a fingernail. But on its own, that sliver is useless. You cannot solder it to
a circuit board, you cannot touch it, and a single speck of dust or drop of moisture can kill it.

Between the bare chip and the green circuit board in your phone sits a step most people never hear
about: **packaging**. And inside most packages, the electrical connection between the chip and the
outside world is made by something surprisingly old-fashioned — **tiny metal wires, thinner than a
human hair, welded one by one by a machine.**

That process is called **wire bonding**, and this series is about how it fails — and how AI can
help catch those failures. But first, the basics.

---

## 1. From wafer to die

Chips are made many at a time on a round slice of silicon called a **wafer** (typically 200 mm or
300 mm across). A single wafer carries hundreds to thousands of identical chips.

```
        WAFER (top view)                     ONE DIE (top view)
     ___________________               ┌────────────────────────┐
   /  □ □ □ □ □ □ □ □   \              │ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ │ ← bond pads
  │ □ □ □ □ □ □ □ □ □ □  │             │ ▪                    ▪ │   along the edge
  │ □ □ □ □ □ □ □ □ □ □  │   ──►       │ ▪   active circuits  ▪ │
  │ □ □ □ □ □ □ □ □ □ □  │   saw       │ ▪   (transistors,    ▪ │
   \  □ □ □ □ □ □ □ □   /              │ ▪    wiring layers)  ▪ │
     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾               │ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ │
                                       └────────────────────────┘
```

After testing, the wafer is sawn apart. Each individual chip is now called a **die** (plural:
dice or dies).

The die's only doors to the outside world are its **bond pads** — small metal squares, usually
aluminium, arranged along the edges of the die. Every signal, every bit of power and every ground
connection must pass through them.

**How small are they?**
- Pads are placed at a **pitch** (centre-to-centre spacing) of roughly **35–100 µm** in modern
  packages; "fine pitch" is often taken as ≤ 100 µm and "ultra-fine pitch" as ≤ 60 µm.
- For comparison, a human hair is about **70–80 µm** thick.

So a whole row of bond pads can fit in the width of a few hairs. You cannot solder to that.

---

## 2. What a "package" actually does

The package is the chip's house. It has four jobs:

| Job | What it means in practice |
|---|---|
| **Connect** | Carry signals and power from the tiny bond pads to large pins or solder balls that can be soldered to a board |
| **Protect** | Shield the fragile die from moisture, dust, light, chemicals and handling |
| **Support** | Give the brittle silicon a rigid mechanical body |
| **Cool** | Conduct heat away from the die |

In this series we care most about the first job — **connect** — because that is where the wires live.

---

## 3. Inside a typical wire-bonded package

Here is a simplified cross-section of a common package (for example a QFN — "Quad Flat No-lead"):

```
  ┌────────────────── mould compound (black epoxy body) ──────────────────┐
  │                                                                        │
  │                        ╭───────── bond wire ─────────╮                │
  │                        │                              ╲               │
  │           ball bond →  ●                                ▸ ← stitch bond│
  │   ┌──────────────────────────────────┐          ┌──────────┐          │
  │   │               DIE                │          │   LEAD   │          │
  │   └──────────────────────────────────┘          └──────────┘          │
  │   ════════ die attach (glue) ═════════                                │
  │   ┌──────────────────────────────────┐                                │
  │   │             DIE PAD              │                                │
  └───┴──────────────────────────────────┴────────────────────────────────┘
            ↑ die pad soldered to the board          ↑ lead soldered to the board
```
*(Simplified one-side view. Every bond pad on the die gets its own wire to its own lead — a
chip with 48 pins typically has around 48 wires like this one — sometimes more, because power and
ground pins often get several wires in parallel.)*

The parts, from the inside out:

1. **Die** — the silicon chip with its bond pads.
2. **Die attach** — an adhesive or solder layer that glues the die to its seat.
3. **Die pad / substrate** — what the die sits on. In leadframe packages (QFN, SOIC, QFP) this is a
   stamped or etched copper frame; in BGA packages it is a small multilayer circuit board.
4. **Leads** (or substrate fingers) — the larger metal areas that eventually connect to the
   circuit board.
5. **Bond wires** — the bridges from each bond pad on the die to its matching lead.
6. **Mould compound** — a black epoxy that is injected around everything and hardens into the
   package body you see on a circuit board.

The wire bond is the **only** electrical path between the die and the leads. One bad wire = one
dead pin = (usually) one dead product.

---

## 4. Why use wires at all?

Because wires solve a size mismatch elegantly and cheaply:

- **Bridging scales.** A bond pad is tens of microns wide; a package lead is hundreds of microns.
  A thin wire spans that jump in one step.
- **Flexibility.** The same bonding machine can connect almost any die to almost any package just by
  changing a software program — no custom tooling per product.
- **Maturity and cost.** Wire bonding has been refined for decades. Industry commentary puts
  flip-chip (the main alternative, below) at roughly 1.4× to 5× the cost of wire bonding, and
  wire bonding is commonly estimated to still make up the large majority of chip interconnects —
  one industry report puts it at about 75–80 %. (These are estimates, not audited figures.)

Wire bonding dominates in microcontrollers, analogue and power chips, sensors, memory, automotive
and most consumer electronics.

### The alternative: flip chip

In **flip-chip** packaging, the die is turned upside down and connected through tiny solder bumps
directly onto the substrate — no wires. It gives shorter, faster connections and is used for
high-performance processors and GPUs. But it is more expensive, so for the bulk of chips made every
year, wires remain the answer.

---

## 5. The wire itself

| Material | Why it's used | Trade-off |
|---|---|---|
| **Gold (Au)** | The classic choice — soft, doesn't oxidise, very forgiving to bond | Expensive |
| **Copper (Cu)** | Much cheaper, conducts better — now the majority of bonding-wire demand | Harder (stresses fragile pads), oxidises (needs a protective gas while bonding) |
| **Silver alloys (Ag)** | Middle ground in cost | Reliability depends on alloy and environment |
| **Aluminium (Al)** | Used for wedge bonding, especially thick wires in power modules | Different bonding style (wedge, not ball) |

Typical diameters for ball bonding are about **15–50 µm** (some sources go up to 75 µm);
fine-pitch products push down to **15–18 µm**. Put simply: **the wire is often thinner than a
human hair, and it must survive moulding, heating cycles and years of use.**

---

## 6. One wire, three places to fail

Every bond wire has three regions. Keep these in mind — the whole defect catalogue in Part 5 is
organised around them:

```
      1st bond              the loop               2nd bond
   (BALL bond on die)    (the wire itself)     (STITCH bond on lead)

        ╭─────────────────────────────────────╮
        │                                      ╲
        ●                                        ▸━━━━
   ┌─────────┐                                ┌────────┐
   │   DIE   │                                │  LEAD  │
```

1. **The ball bond** — where the wire is welded to the die's bond pad.
2. **The loop** — the arched span of wire through the air (later, through mould compound).
3. **The stitch bond** — where the wire is pressed and cut onto the lead.

Most defects you will see under a microscope or an AOI camera — lifted balls, wire sweep, broken
wires, heel cracks — happen at one of these three places.

---

## 7. Why defects here are expensive

- Bonding happens **late** in manufacturing: the silicon is already made, tested and paid for.
  A defect here throws away everything that came before.
- Some defects are **invisible later**: once the black mould compound is on, you can only see the
  wires with **X-ray**. That is why inspection right after bonding (microscope, AOI) matters.
- Some defects are **latent**: a weak bond may pass every test today and fail in the field months
  later after thousands of heat cycles — in a car, a medical device or a satellite.

That is the motivation for this series: understand the defects precisely, separate the **random**
ones from the **repetitive** ones (Part 6), and then see how AI can help inspectors catch them
faster and more consistently (Part 7).

---

## Key terms (glossary)

| Term | Meaning |
|---|---|
| **Wafer** | The silicon disc on which many chips are made at once |
| **Die** | One individual chip cut from the wafer |
| **Bond pad** | Metal pad on the die where a wire is attached |
| **Pitch** | Centre-to-centre distance between neighbouring pads |
| **Leadframe** | Metal frame that holds the die and provides the package leads |
| **Substrate** | Small circuit board used as the base in BGA-type packages |
| **Die attach** | Adhesive/solder that fixes the die in place |
| **Mould compound** | Black epoxy that encapsulates the package |
| **Ball bond (1st bond)** | Wire-to-die weld, formed from a melted ball of wire |
| **Stitch bond (2nd bond)** | Wire-to-lead weld, made by pressing and cutting the wire |
| **Flip chip** | Wire-free alternative using solder bumps |
| **AOI** | Automated Optical Inspection — camera-based inspection |

---

## Next: Part 2 — How wire bonding works

How does a machine weld a wire thinner than a hair, thousands of times per hour, without
melting the chip? Capillaries, free-air balls, ultrasonic energy — next time.

---

### Sources

- Pad pitch definitions (fine ≤ 100 µm, ultra-fine ≤ 60 µm): US patent 11901327, *Wire bonding for
  semiconductor devices* — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11901327
- Pad pitch examples (50 µm, 90 µm dies): US patent 6563226, *Bonding pad* —
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6563226
- Ball-bond pitch and wire diameters: LCSC, *Wire Bonding Pad Design* —
  https://www.lcsc.com/blog/?p=4177
- 15 µm gold wire at 35 µm pitch: Tanaka Precious Metals, *Gold and Gold Alloy Bonding Wires* —
  https://tanaka-preciousmetals.com/en/?p=4455137
- 18 µm silver-alloy wire at 40 µm pitch; copper hardness trade-off: Amkor QFN datasheet —
  https://www.alldatasheet.jp/html-pdf/2018639/AMKOR/QFN/170/1/QFN.html
- Wire bonding ~75–80 % of interconnects (estimate): PatSnap Eureka —
  https://eureka.patsnap.com/report-wire-bonding-vs-flip-chip-semiconductor-packaging-efficiency
- Flip chip cost 140–400 % more than wire bonding (K&S CEO via EE Times) —
  https://www.eetimes.com/?p=1189921
- Wire bonding market ~$5B/yr: SemiconductorX — https://semiconductorx.com/packaging-wire-bonding.html
- Copper ~68 % of bonding-wire demand: openPR / DataHorizzon —
  https://www.openpr.com/news/4590355/bonding-wire-packaging-material-market-size-to-reach-usd-4-9
- Wire bonding vs flip chip overview: JLCPCB — https://jlcpcb.com/blog/wire-bonding-vs-flip-chip
