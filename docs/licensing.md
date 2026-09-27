# Project licensing

On 2026-09-27, the maintainer adopted the
[Gestur Noncommercial Attribution License 1.0](../LICENSE). It applies from the
Git revision that introduces version 1.0 and to later revisions distributed with
it. The license text governs; this guide explains its scope and operation.

## Permitted use and commercial permission

Original Gestur code, documentation, and configuration may be used, studied,
modified, and shared without charge for noncommercial purposes under the license.
Commercial use requires prior, explicit written permission from **Xavier Burgos**.
Contact [xavi@dzin.es](mailto:xavi@dzin.es) with the proposed use.

Commercial purposes include business operations, advertising, paid hosting or
installation services, and paid or commercially sponsored exhibitions. Being a
nonprofit or educational institution does not automatically exempt an activity.
The license distinguishes donations without consideration from payments tied to
access, services, or promotion. Attribution alone does not authorize commercial
use. A separate commercial agreement should identify the operator, deployment,
duration, attribution, redistribution rights, and any fee.

## Attribution

Retain the license, copyright notices, project URL, and required attribution in
distributed copies and substantial portions, including modifications:

**Gestur — developed by Xavier Burgos**

Project: https://github.com/xavi-burgos99/gestur

Public installations must show this credit legibly on the display or on a clearly
visible adjacent sign. A source comment or private administration page alone is
not enough for visitors. Distributed web interfaces must retain an accessible,
clearly labelled developer credit; the portal includes one in Configuración.
Private use does not require a public announcement, but existing notices must
be retained. Mark modifications and do not imply the author's endorsement.

## License choice

This is **source-available**, not open source as defined by the Open Source
Initiative. Sections 1 and 6 of the [Open Source Definition](https://opensource.org/osd)
do not allow restricting commercial redistribution or business use. MIT, BSD,
Apache, GPL, and AGPL therefore do not implement this policy.

The adopted custom license makes both commercial permission and visible public
attribution explicit. [PolyForm Noncommercial](https://polyformproject.org/licenses/noncommercial/1.0.0)
was considered, but its copy notices do not require a visible credit in every
public installation, and it separately permits specified institutional uses
regardless of funding. Gestur's text is not presented as the PolyForm license.

CC BY-NC and related licenses are not used for the software:
[Creative Commons advises against its licenses for software](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software).

## Third-party and historical rights

The license covers only rights the licensor owns or is authorized to grant.
Dependencies, OS packages, downloaded model weights, vendored research code,
fonts, and independently licensed assets retain their original terms. It does
not remove MIT, BSD, Apache, LGPL, or other rights from those components. The
[dependency audit](licensing-audit.md) and [third-party notices](../THIRD_PARTY_NOTICES.md)
identify the evidence and obligations, including native libraries in an image.

The capitel mesh, material, texture, and ZIP have been removed from the current
source tree and preserved privately outside the repository. They are not part
of this license or the current distribution. Historical benchmark measurements
and file hashes remain as provenance. Their presence does not grant rights to
the model. This removal does not rewrite earlier Git commits or other branches.

An earlier root Apache-2.0 license appears in the Git history alongside an
OpenMMLab/MMPose import. Its original scope is not established by the copyright
line alone. The new license cannot cancel valid grants for earlier copies or
third-party portions. Adoption also does not certify ownership of every
contribution, university agreement, or previously included asset.

No copyright license can prohibit every conceivable benefit: it covers the
rights owned by the licensor, subject to statutory exceptions. It does not
monopolize an idea or prevent independent implementation. A commercial agreement
for Gestur cannot grant rights over excluded third-party material.

## Distribution and contributions

Distribute `LICENSE`, the required attribution, and applicable third-party
notices with the covered software. A preinstalled Raspberry Pi image also needs
the original OS/native notices and the corresponding-source and library
replacement arrangements required by the exact packages distributed. Installing
dependencies on an end user's device is a different distribution scenario.

External contributions need written terms that authorize any later commercial
licensing by the maintainer; the noncommercial project license alone does not
grant that additional authority. See [CONTRIBUTING.md](../CONTRIBUTING.md).

Adoption records the maintainer's decision, not a completed legal opinion or
clearance of third-party rights. Legal review is appropriate before relying on
the custom terms in a distribution or commercial agreement.

## Primary references

- [Open Source Definition](https://opensource.org/osd)
- [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
- [Creative Commons software guidance](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software)
- [Apache License 2.0, including its irrevocable copyright grant](https://www.apache.org/licenses/LICENSE-2.0)
- [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.html)
